import signal
import sys
import cv2
import csv
import gymnasium as gym
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

SAVE_PATH = "./models"
GAMES = {"space_invaders": "ALE/SpaceInvaders-v5", "breakout": "ALE/Breakout-v5"}
MAX_FRAMES = 50000  # total training frames
MAX_BUFFER = 5000  # size of replay buffer
STEPS_TO_UPDATE = 1000  # steps between target updates
STACKED_FRAMES = 4  # number of frames to consider as observations

# Configuration paramaters for the whole setup
seed = 42


@dataclass
class Stats:
    episodes: List
    scores: List
    steps_survived: List

    def append(self, episode, score, steps):
        self.episodes.append(episode)
        self.scores.append(score)
        self.steps_survived.append(steps)


def make_model(input_shape, n_actions, version=2):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(STACKED_FRAMES, *input_shape))

    match version:
        case 1:
            layer1 = layers.Conv2D(16, 9, strides=4, activation="relu")(inputs)
            layer2 = layers.Conv2D(16, 5, strides=2, activation="relu")(layer1)
            layer3 = layers.Conv2D(16, 3, strides=1, activation="relu")(layer2)
            layer4 = layers.Flatten()(layer3)
            layer5 = layers.Dense(512, activation="relu")(layer4)
            action = layers.Dense(n_actions, activation="linear")(layer5)

        case 2:
            layer1 = layers.Conv2D(16, 8, strides=4, activation="relu")(inputs)
            layer2 = layers.Conv2D(32, 4, strides=2, activation="relu")(layer1)
            layer3 = layers.Flatten()(layer2)
            layer4 = layers.Dense(256, activation="relu")(layer3)
            action = layers.Dense(n_actions, activation="linear")(layer4)

    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(keras.optimizers.Adam(learning_rate=0.001), keras.losses.Huber())
    return model


def save(model, name, version):
    model.save(f"{SAVE_PATH}/{name}_v{version}", include_optimizer=False)


def save_stats(stats: Stats, name, version):
    fields = list(stats.__dict__.keys())
    with open(f"{SAVE_PATH}/{name}_v{version}/stats.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(np.array(list(stats.__dict__.values())).T)


def plot_stats(name: str, version: int):
    df = pd.read_csv(f"./models/{name}_v{version}/stats.csv")
    fig, ax = plt.subplots(2)
    ax[0].plot(df.episode, df.score)
    ax[1].plot(df.episode, df.steps_survived)
    plt.show()


def load(name: str):
    return tf.keras.models.load_model(f"{SAVE_PATH}/{name}")


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    if len(img.shape) == 3:
        y_size, x_size, _ = img.shape
        return cv2.resize(img, dsize=(int(x_size // factor), int(y_size // factor)))
    else:
        downscaled = []
        y_size, x_size, _ = img[0].shape
        for frame in img:
            downscaled.append(
                cv2.resize(frame, dsize=(int(x_size // factor), int(y_size // factor)))
            )
        return np.array(downscaled)


def train_batch(policy, target, buffer, batch_size, gamma):
    """Trains a single batch of experiences from the buffer."""
    transitions = np.array(
        [buffer[i] for i in np.random.choice(len(buffer), batch_size, replace=False)],
        dtype=object,
    )
    (frames, actions, rewards, next_frames, is_terminated) = tuple(
        [np.array(transitions[:, i]) for i in range(len(transitions[0]))]
    )
    frames = np.array([downscale(f, 2) for f in frames])
    next_frames = np.array([downscale(next_f, 2) for next_f in next_frames])
    next_q_values = target(next_frames).numpy()
    max_next_q_values = np.max(next_q_values, axis=1)
    target_q_values = rewards + (1 - is_terminated) * gamma * max_next_q_values

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.Huber()
    mask = tf.one_hot(actions, len(next_q_values[0]))
    with tf.GradientTape() as tape:
        raw_q_values = policy(frames)
        q_values = tf.reduce_sum(raw_q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q_values.astype("float32"), q_values))
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))


def synchronize_model(policy, target):
    """Synchronizes weights between policy and target."""
    target.set_weights(policy.get_weights())


def epsilon_greedy(policy, state, epsilon: float) -> int:
    """Returns best action following an epsilon-greedy policy."""
    q_values = policy(np.array([downscale(np.array(state), 2)]))[0].numpy()
    e = np.random.rand()
    if e <= epsilon:
        # Select random action
        action = np.random.randint(len(q_values))
    else:
        # Select action with highest q-value
        action = np.argmax(q_values)
    return action


def init_stacked_frames(observation: np.ndarray, max_frames):
    frames = deque(maxlen=max_frames)
    for _ in range(max_frames):
        frames.append(observation)
    next_frames = deepcopy(frames)
    return frames, next_frames


def visualize(frames):
    fig, ax = plt.subplots(2, STACKED_FRAMES, figsize=(15, 15))
    for j in range(STACKED_FRAMES):
        ax[0][j].imshow(frames[j])
        ax[0][j].axis("off")
    for j in range(STACKED_FRAMES):
        ax[1][j].imshow(downscale(frames[j], 2))
        ax[1][j].axis("off")
    plt.savefig("./current_frame.png")


def train(
    name: str,
    version: int = 2,
    render: bool = False,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    exploration_frames: int = MAX_FRAMES / 10,
    batch_size: int = 32,
    steps_to_update: int = 4,
    debug: bool = False,
):
    """
    Trains the model.

    Arguments:
        name: Name of the game.
        version: Model version.
        render: Whether to render training or not.
        gamma: Discount factor.
        epsilon: Epsilon greedy parameter.
        exploration_frames: Number of frames to reach minimum epsilon.
        batch_size: Size of experience replay batch.
        steps_to_update: Number of steps between batch training.
        debug: Visual debugging.
    """

    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    # Creating the environment
    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human" if render else None)
    env.seed(seed)
    observation, info = env.reset(seed=seed)
    frames, next_frames = init_stacked_frames(observation, STACKED_FRAMES)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    # Defining the Q-learning parameters
    epsilon_decay = 0.9 / exploration_frames
    step_count = 0
    replay_buffer = deque(maxlen=MAX_BUFFER)

    # Initialising the model
    n_actions = int(env.action_space.n)
    space_shape = downscale(observation, 2).shape
    target = make_model(space_shape, n_actions, version)
    policy = deepcopy(target)
    steps_to_synchronize = 200

    # Variables to perform analysis
    stats = Stats([], [], [])
    running_reward = 0
    steps_survived = 0
    episode = 1

    # Training
    for _ in tqdm(range(MAX_FRAMES)):
        step_count += 1
        steps_survived += 1

        action = epsilon_greedy(policy, frames, epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        running_reward += reward
        next_frames.append(next_observation)
        replay_buffer.append(
            (np.array(frames), action, reward, np.array(next_frames), terminated)
        )
        frames.append(next_observation)

        if terminated:
            stats.append(episode, running_reward, steps_survived)
            running_reward = 0
            steps_survived = 0
            episode += 1
            observation, info = env.reset(seed=seed)
            frames, next_frames = init_stacked_frames(observation, STACKED_FRAMES)
            continue

        if step_count % steps_to_update == 0 and len(replay_buffer) > batch_size:
            train_batch(policy, target, replay_buffer, batch_size, gamma)

        if step_count % steps_to_synchronize == 0:
            synchronize_model(policy, target)

        if debug:
            visualize(frames)

        epsilon -= epsilon_decay

    # Save results
    save(policy, name, version)
    save_stats(stats, name, version)


def test(name: str, version: Optional[int] = None, episodes: int = -1, render=True):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human" if render else None)
    model = load(f"{name}_v{version}") if version is not None else None
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    score = 0
    i = 0
    while i != episodes:
        i += 1
        observation, _ = env.reset()
        terminated = False
        while not terminated:
            if model is not None:
                action = np.argmax(model(np.array([downscale(observation, 2)])))
            else:
                action = np.random.randint(env.action_space.n)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
    return score


def compare(
    episodes: int,
    name: str,
    player_1: int,
    player_2: Optional[int] = None,
    render=False,
):
    """Compare total model scores over a number of episodes."""
    score_1 = test(name, player_1, episodes, render)
    score_2 = test(name, player_2, episodes, render)
    print(f"Player 1: {score_1}\tPlayer 2: {score_2}")


if __name__ == "__main__":
    # compare(25, "space_invaders", 1)
    # plot_stats("space_invaders", 2)
    train("space_invaders", 2, render=False)
    # test("space_invaders", 3)
    # import matplotlib.pyplot as plt
    # env = gym.make(GAMES["space_invaders"], obs_type="rgb")
    # env.seed(42)
    # obs, info = env.reset(seed=42)
    # space_shape = downscale(obs, 2).shape
    # fig, ax = plt.subplots(1, 2)
    # print(space_shape)

    # ax[0].imshow(obs)
    # res = downscale(obs, 2)
    # ax[1].imshow(res)
    # plt.show()
