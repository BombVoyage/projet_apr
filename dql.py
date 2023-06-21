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

SAVE_PATH = "./models"
GAMES = {"space_invaders": "ALE/SpaceInvaders-v5", "breakout": "ALE/Breakout-v5"}
MAX_FRAMES = 50000  # total training frames
MAX_BUFFER = 5000  # size of replay buffer
STEPS_TO_UPDATE = 1000  # steps between target updates

# Configuration paramaters for the whole setup
seed = 42


@dataclass
class Stats:
    episode: np.ndarray
    score: np.ndarray
    steps_survived: np.ndarray


def make_model(input_shape, n_actions, version=2):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=input_shape)

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
    y_size, x_size, _ = img.shape
    return cv2.resize(img, dsize=(int(x_size // factor), int(y_size // factor)))


def train(name: str, version: int = 2, render: bool = False):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    # Creating the environment
    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human" if render else None)
    env.seed(seed)
    observation, info = env.reset(seed=seed)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    # Defining the Q-learning parameters
    gamma = 0.9  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    exploration_frames = MAX_FRAMES / 10
    epsilon_decay = 0.9 / exploration_frames
    batch_size = 32
    steps_to_update = 4
    step_count = 0
    replay_buffer = deque(maxlen=MAX_BUFFER)

    # Initialising the model
    n_actions = int(env.action_space.n)
    space_shape = downscale(observation, 2).shape
    target = make_model(space_shape, n_actions, version)
    policy = deepcopy(target)
    steps_to_synchronize = 200

    # Variables to perform analysis
    stats = Stats(np.array([0]), np.array([0]), np.array([0]))
    running_reward = 0
    steps_survived = 0

    for _ in tqdm(range(MAX_FRAMES)):
        step_count += 1
        q_values = policy(np.array([downscale(observation, 2)]))[0].numpy()

        e = np.random.rand()
        if e <= epsilon:
            # Select random action
            action = np.random.randint(n_actions)
        else:
            # Select action with highest q-value
            action = np.argmax(q_values)

        next_observation, reward, terminated, truncated, info = env.step(action)
        running_reward += reward
        steps_survived += 1
        # Save to replay buffer
        replay_buffer.append(
            (observation, action, reward, next_observation, terminated)
        )

        if terminated:
            stats.episode = np.append(stats.episode, stats.episode[-1] + 1)
            stats.score = np.append(stats.score, running_reward)
            stats.steps_survived = np.append(stats.steps_survived, steps_survived)
            running_reward = 0
            steps_survived = 0
            observation, info = env.reset(seed=seed)
            continue

        if step_count % steps_to_update == 0 and len(replay_buffer) > batch_size:
            transitions = np.array(
                [
                    replay_buffer[i]
                    for i in np.random.choice(
                        len(replay_buffer), batch_size, replace=False
                    )
                ]
            )
            (observations, actions, rewards, next_observations, is_terminated) = tuple(
                [np.array(transitions[:, i]) for i in range(len(transitions[0]))]
            )
            observations = np.array([downscale(obs, 2) for obs in observations])
            next_observations = np.array(
                [downscale(next_obs, 2) for next_obs in next_observations]
            )
            max_next_q_values = np.max(target(next_observations).numpy(), axis=1)
            target_q_values = rewards + (1 - is_terminated) * gamma * max_next_q_values

            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            loss_fn = keras.losses.Huber()
            mask = tf.one_hot(actions, n_actions)
            with tf.GradientTape() as tape:
                raw_q_values = policy(observations)
                q_values = tf.reduce_sum(raw_q_values * mask, axis=1, keepdims=True)
                loss = tf.reduce_mean(
                    loss_fn(target_q_values.astype("float32"), q_values)
                )
            grads = tape.gradient(loss, policy.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        if step_count % steps_to_synchronize == 0:
            target.set_weights(policy.get_weights())

        epsilon -= epsilon_decay
    save(policy, name, version)
    save_stats(stats, name, version)


def test(name: str, version: int):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human")
    model = load(f"{name}_v{version}")
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))
    while True:
        observation, _ = env.reset()
        terminated = False
        while not terminated:
            action = np.argmax(model(np.array([downscale(observation, 2)])))
            observation, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    # plot_stats("space_invaders", 2)
    train("space_invaders", 2, render="human")
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
