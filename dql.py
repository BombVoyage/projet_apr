import argparse
import csv
import logging
import signal
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium.wrappers import FrameStack
from skimage import color
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

SAVE_PATH = "./models"
GAMES = {
    "space_invaders": "SpaceInvadersDeterministic-v4",
    "breakout": "ALE/Breakout-v5",
    "freeway": "ALE/Freeway-v5",
    "pong": "ALE/Pong-v5",
    "pacman": "ALE/Pacman-v5",
}


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
    new_dim = int(max(x_size // factor, y_size // factor))
    return cv2.resize(img, dsize=(new_dim, new_dim))


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
    current_frame_delta = np.maximum(
        frames[..., 3] - frames[..., :3].mean(axis=-1), 0.0
    )
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    return img


def synchronize_model(policy, target):
    """Synchronizes weights between policy and target."""
    target.set_weights(policy.get_weights())
    logging.info("Models synchronized.")


def epsilon_greedy(policy, state, epsilon: float) -> int:
    """Returns best action following an epsilon-greedy policy."""
    e = np.random.rand()
    if e <= epsilon:
        # Select random action
        action = np.random.randint(len(policy.get_weights()[-1]))
    else:
        # Select action with highest q-value
        q_values = policy(state, training=False).numpy()
        action = np.argmax(q_values)
        logging.info(f"\nMeilleure action: {action}" f"\nQ Values: {q_values}")
    return action


def merge_frames(frames, to_preprocess: bool = False):
    if to_preprocess:
        frames = np.array([preprocess(f) for f in frames])
    else:
        frames = np.array(frames)

    if len(frames.shape) == 4:
        return 1 - apply_delta(np.concatenate(tuple(frames), axis=2))
    elif len(frames.shape) == 5:
        return np.array([merge_frames(f) for f in frames])


def visualize(frames):
    """Save a visualization of the current frames given to the model."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    obs = merge_frames(frames, to_preprocess=True)
    ax[0].imshow(frames[-1])
    ax[1].imshow(obs)
    ax[0].axis("off")
    ax[1].axis("off")
    plt.savefig("./images/current_frame.png")
    plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(15, 15))
    obs = merge_frames(frames, to_preprocess=True)
    ax[0].imshow(obs * np.array([1, 0, 0]))
    ax[1].imshow(obs * np.array([0, 1, 0]))
    ax[2].imshow(obs * np.array([0, 0, 1]))
    ax[3].imshow(obs * np.array([1, 0, 1]))
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")
    plt.savefig("./images/current_rgb.png")
    plt.close()


def visualize_batch(buffer, batch_size):
    """Sample and visualize a random batch from the buffer."""
    i = np.random.randint(0, len(buffer) - batch_size)
    observations = np.array(
        [merge_frames(buffer[i + k][0], to_preprocess=True) for k in range(batch_size)]
    )
    n = int(np.ceil(batch_size**0.5))
    fig, ax = plt.subplots(n, n, figsize=(15, 15))
    for i in range(n * n):
        ax[i // n][i % n].axis("off")
    for i, obs in enumerate(observations):
        ax[i // n][i % n].imshow(obs)
        ax[i // n][i % n].title.set_text(i)
    plt.subplots_adjust(wspace=0.01, hspace=0.2)
    plt.savefig("./images/example_batch.png")
    plt.close()


def train_batch(policy, target, buffer, batch_size, gamma):
    """Trains a single batch of experiences from the buffer."""
    i = np.random.randint(0, len(buffer) - batch_size)
    observations = np.array(
        [merge_frames(buffer[i + k][0], to_preprocess=True) for k in range(batch_size)]
    )
    actions = np.array([buffer[i + k][1] for k in range(batch_size)])
    rewards = np.array([buffer[i + k][2] for k in range(batch_size)])
    next_observations = np.array(
        [merge_frames(buffer[i + k][3], to_preprocess=True) for k in range(batch_size)]
    )
    is_terminated = np.array([buffer[i + k][4] for k in range(batch_size)])

    max_next_q_values = tf.reduce_max(
        target.predict(next_observations, verbose=False), axis=1
    )
    target_q_values = rewards + (1 - is_terminated) * gamma * max_next_q_values

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.Huber()
    mask = tf.one_hot(actions, len(policy.get_weights()[-1]))
    with tf.GradientTape() as tape:
        raw_q_values = policy(observations)
        q_values = tf.reduce_sum(tf.multiply(raw_q_values, mask), axis=1)
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    # Log for debugging
    logging.info(
        "\n================= BATCH START =================\n"
        # f"Next Q\n{next_q_values}\n"
        # f"Best next actions\n{best_next_actions}\n"
        # f"Max next Q\n{max_next_q_values}\n"
        f"Target Q\n{target_q_values}\n"
        f"Current Q\n{q_values.numpy().reshape((batch_size,))}\n"
        f"Actions\n{actions}\n"
        f"Rewards\n{rewards}\n"
        f"Not terminated\n{1-is_terminated}\n"
        f"Loss\n{loss}"
        "\n================= BATCH END =================\n"
    )


def train(
    name: str,
    version: int = 2,
    render: bool = False,
    max_frames: int = 50000,
    max_buffer: int = 5000,
    stacked_frames: int = 4,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    exploration_time: int = 1 / 10,
    batch_size: int = 32,
    steps_to_update: int = 4,
    steps_to_synchronize: int = 10000,
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
    env = FrameStack(env, stacked_frames, lz4_compress=True)
    observation, info = env.reset()
    lives = info["lives"]
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    # Defining the Q-learning parameters
    exploration_frames = exploration_time * max_frames
    epsilon_decay = 0.9 / exploration_frames
    replay_buffer = deque(maxlen=max_buffer)

    # Initialising the model
    n_actions = int(env.action_space.n)
    target = make_model(
        merge_frames(observation, to_preprocess=True), n_actions, version
    )
    policy = make_model(
        merge_frames(observation, to_preprocess=True), n_actions, version
    )

    # Variables to perform analysis
    stats = Stats([], [], [])
    running_reward = 0
    steps_survived = 0
    episode = 1

    # Training
    for step_count in tqdm(range(1, max_frames + 1)):
        steps_survived += 1
        if info["lives"] != lives:
            env.step(1)  # Needed to start some games (like breakout)
            lives = info["lives"]

        action = epsilon_greedy(
            policy, np.array([merge_frames(observation, to_preprocess=True)]), epsilon
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        running_reward += reward
        replay_buffer.append(
            (observation, action, reward, next_observation, terminated)
        )
        observation = next_observation

        if terminated:
            stats.append(episode, running_reward, steps_survived)
            running_reward = 0
            steps_survived = 0
            episode += 1
            observation, info = env.reset()
            print(f"Avg score: {np.mean(stats.scores)}")
            continue

        if step_count % steps_to_update == 0 and len(replay_buffer) > batch_size:
            train_batch(policy, target, replay_buffer, batch_size, gamma)
        if step_count % steps_to_synchronize == 0:
            synchronize_model(policy, target)
        if step_count % steps_to_checkpoint == 0:
            save_checkpoint(policy, name, version)

        if debug and step_count % 50 == 0:
            visualize(observation)
        if debug and step_count % (batch_size * 5) == 0:
            visualize_batch(replay_buffer, batch_size)

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
    env = FrameStack(env, stacked_frames, lz4_compress=True)
    model = load(f"{name}_v{version}") if version is not None else None
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    score = 0
    i = 0
    lives = -1
    while i != episodes:
        i += 1
        observation, info = env.reset()
        terminated = False
        while not terminated:
            if info["lives"] != lives:
                env.step(1)  # Needed to start some games (like breakout)
                lives = info["lives"]
            if model is not None:
                action = epsilon_greedy(
                    model,
                    np.array([merge_frames(observation, to_preprocess=True)]),
                    epsilon,
                )
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
    modes = ["train", "test", "plot", "compare"]
    defaults = {
        "steps": 100000,
        "buffer": 5000,
        "gamma": 0.99,
        "epsilon": 1.0,
        "exploration": 1 / 10,
    }

    parser = argparse.ArgumentParser(
        description="Train and test DQN models on atari games."
    )
    parser.add_argument(
        "mode", metavar="mode", type=str, nargs=1, help=f"Mode of use. {modes}"
    )
    parser.add_argument(
        "game", metavar="game", type=str, nargs=1, help="Name of the game."
    )
    parser.add_argument(
        "version",
        metavar="version",
        type=int,
        nargs=1,
        help="Version of the neural network.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        help=f'Total training steps. Default: {defaults["steps"]}',
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=int,
        help=f'Size of replay buffer. Default: {defaults["buffer"]}',
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        help=f'Gamma parameter for Q learning. Default: {defaults["gamma"]}',
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        help=f'Initial parameter for epsilon greedy policy. Default: {defaults["epsilon"]}',
    )
    parser.add_argument(
        "-x",
        "--exploration",
        type=float,
        help=f'Fraction of total steps before epsilon reaches minimum value (0.1). Default: {defaults["exploration"]}',
    )
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")

    args = parser.parse_args()
    mode = args.mode[0].lower()
    name = args.game[0]
    version = args.version[0]
    render = args.render
    debug = args.debug

    if mode in modes and args.log:
        logging.basicConfig(
            filename=f"./logs/dql_{mode}.log", filemode="w", level=logging.INFO
        )
    else:
        print(f"Unrecognized mode {mode}.\n")

    if mode == "train":
        train(
            name,
            version,
            render=render,
            stacked_frames=4,
            batch_size=32,
            max_frames=args.steps or defaults["steps"],
            max_buffer=args.buffer or defaults["buffer"],
            gamma=args.gamma or defaults["gamma"],
            epsilon=args.epsilon or defaults["epsilon"],
            exploration_time=args.exploration or defaults["exploration"],
            debug=debug,
        )
    if mode == "test":
        test(name, version, epsilon=0.05)
    if mode == "plot":
        plot_stats(name, version)
    if mode == "compare":
        compare(25, name, version)
