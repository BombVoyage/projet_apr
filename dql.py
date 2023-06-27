import signal
import sys
import cv2
import csv
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage import color
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

SAVE_PATH = "./models"
GAMES = {"space_invaders": "SpaceInvadersDeterministic-v4", "breakout": "ALE/Breakout-v5"}

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


def make_model(sample_input: np.ndarray, n_actions, version=2):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=sample_input.shape)

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

        case 3:
            layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
            layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
            layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
            layer4 = layers.Flatten()(layer3)
            layer5 = layers.Dense(512, activation="relu")(layer4)
            action = layers.Dense(n_actions, activation="linear")(layer5)

    model = keras.Model(inputs=inputs, outputs=action)
    return model


def save(model, name, version):
    model.save(f"{SAVE_PATH}/{name}_v{version}", include_optimizer=False)


def save_stats(stats: Stats, name, version):
    fields = list(stats.__dict__.keys())
    with open(f"{SAVE_PATH}/{name}_v{version}/stats.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(np.array(list(stats.__dict__.values())).T)


def save_checkpoint(model, name: str, version: int):
    model.save_weights(f"{SAVE_PATH}/{name}_v{version}/checkpoint/checkpoint")


def plot_stats(name: str, version: int):
    df = pd.read_csv(f"./models/{name}_v{version}/stats.csv")
    fig, ax = plt.subplots(2)
    ax[0].plot(df.episodes, df.scores)
    ax[1].plot(df.episodes, df.steps_survived)
    plt.show()


def load(name: str):
    return tf.keras.models.load_model(f"{SAVE_PATH}/{name}")


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    y_size, x_size = img.shape[0], img.shape[1]
    return cv2.resize(img, dsize=(int(x_size // factor), int(y_size // factor)))


def grayscale(img: np.ndarray):
    return color.rgb2gray(img)


def preprocess(img, downscale_factor: float = 2):
    img = np.array(img)
    if len(img.shape) == 3:
        res = downscale(grayscale(img), downscale_factor)
        res = res.reshape((*res.shape, 1))
        return res
    else:
        return np.array([preprocess(frame) for frame in img])


def apply_delta(frames):
    img = frames[..., :3]
    current_frame_delta = np.maximum(frames[..., 3] - frames[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img/np.mean(img), 0, 1)
    return img


def train_batch(policy, target, buffer, batch_size, gamma):
    """Trains a single batch of experiences from the buffer."""
    batch = random.sample(list(buffer), batch_size)
    frames = np.array([merge_frames(b[0]) for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.array([b[2] for b in batch])
    next_observations = np.array([b[3] for b in batch])
    is_terminated = np.array([b[4] for b in batch])

    next_frames = np.array(
        [
            np.concatenate((frames[i][:, :, 1:], next_observations[i]), axis=2)
            for i in range(batch_size)
        ]
    )
    next_q_values = policy(next_frames).numpy()
    best_next_actions = np.argmax(next_q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, len(next_q_values[0])).numpy()
    max_next_q_values = (target(next_frames).numpy()*next_mask).sum(axis=1)
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
    q_values = policy(state).numpy()
    e = np.random.rand()
    if e <= epsilon:
        # Select random action
        action = np.random.randint(len(q_values[0]))
    else:
        # Select action with highest q-value
        action = np.argmax(q_values)
        print(f"\nMeilleure action: {action}")
    return action


def init_stacked_frames(
    observation: np.ndarray, stacked_frames
):
    """Create the initial stacked frames."""
    frames = deque(maxlen=stacked_frames)
    for _ in range(stacked_frames):
        frames.append(observation)
    return frames


def merge_frames(frames):
    frames = np.array(frames)
    if len(frames.shape) == 4:
        return np.concatenate(tuple(frames), axis=2)
    elif len(frames.shape) == 5:
        return np.array([merge_frames(f) for f in frames])


def visualize(frames):
    """Save a visualization of the current frames given to the model."""
    fig, ax = plt.subplots(1, figsize=(15, 15))
    obs = merge_frames(frames)
    img = apply_delta(obs)
    ax.imshow(img)
    ax.axis("off")
    plt.savefig("./current_frame.png")
    plt.close()


def train(
    name: str,
    version: int = 2,
    render: bool = False,
    max_frames: int = 50000,
    max_buffer: int = 5000,
    stacked_frames: int = 4,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    exploration_time: int = 1 / 10,
    batch_size: int = 32,
    steps_to_update: int = 4,
    steps_to_synchronize: int = 200,
    steps_to_checkpoint: int = 10000,
    debug: bool = False,
):
    """
    Trains the model.

    Arguments:
        name -- Name of the game.
        version -- Model version.
        render -- Whether to render training or not.
        max_frames -- Maximum training frames.
        max_buffer -- Maximum buffer size for experience replay.
        stacked_frames -- Number of frames constituting a single state.
        gamma -- Discount factor.
        epsilon -- Epsilon greedy parameter.
        exploration_time -- Fraction of max frames before minimum epsilon is reached.
        batch_size -- Size of experience replay batch.
        steps_to_update -- Number of steps between batch training.
        steps_to_synchronize -- Number of steps between model synchronization.
        steps_to_checkpoint -- Number of steps between model checkpoints.
        debug -- Visual debugging.
    """

    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    # Creating the environment
    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human" if render else None)
    env.seed(seed)
    observation, info = env.reset(seed=seed)
    frames = init_stacked_frames(preprocess(observation), stacked_frames)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    # Defining the Q-learning parameters
    exploration_frames = exploration_time * max_frames
    epsilon_decay = 0.9 / exploration_frames
    replay_buffer = deque(maxlen=max_buffer)

    # Initialising the model
    n_actions = int(env.action_space.n)
    target = make_model(merge_frames(frames), n_actions, version)
    policy = deepcopy(target)

    # Variables to perform analysis
    stats = Stats([], [], [])
    running_reward = 0
    steps_survived = 0
    episode = 1

    # Training
    for step_count in tqdm(range(max_frames)):
        steps_survived += 1

        action = epsilon_greedy(policy, np.array([merge_frames(frames)]), epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        running_reward += reward
        replay_buffer.append(
            (np.array(frames), action, reward, preprocess(next_observation), terminated)
        )
        frames.append(preprocess(next_observation))

        if terminated:
            stats.append(episode, running_reward, steps_survived)
            running_reward = 0
            steps_survived = 0
            episode += 1
            env.reset(seed=seed)
            observation, _, _, _, _ = env.step(1)  # Needed to start some games, like breakout
            frames = init_stacked_frames(preprocess(observation), stacked_frames)
            continue

        if step_count % steps_to_update == 0 and len(replay_buffer) > batch_size:
            train_batch(policy, target, replay_buffer, batch_size, gamma)
        if step_count % steps_to_synchronize == 0:
            synchronize_model(policy, target)
        if step_count % steps_to_checkpoint == 0:
            save_checkpoint(policy, name, version)
        if debug and step_count%50==0:
            visualize(frames)

        if step_count < exploration_frames:
            epsilon -= epsilon_decay

    # Save results
    save(policy, name, version)
    save_stats(stats, name, version)


def test(
    name: str,
    version: Optional[int] = None,
    epsilon: float = 0,
    episodes: int = -1,
    render=True,
    stacked_frames: int = 4,
):
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
        env.reset()
        observation, _, _, _, _ = env.step(1)  # Needed to start some games, like breakout
        frames = init_stacked_frames(preprocess(observation), stacked_frames)
        terminated = False
        while not terminated:
            if model is not None:
                action = epsilon_greedy(model, np.array([merge_frames(frames)]), epsilon)
                print(action)
            else:
                action = np.random.randint(env.action_space.n)
            observation, reward, terminated, truncated, info = env.step(action)
            frames.append(preprocess(observation))
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
    name = "space_invaders"
    # compare(25, "space_invaders", 2)
    train(name, 3, render=False, stacked_frames=4, max_frames=100000, debug=False)
    # plot_stats(name, 3)
    # test(name, 3, epsilon=0.05)
