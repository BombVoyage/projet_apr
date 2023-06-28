import signal
import sys
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import csv
from skimage import color
from gymnasium.wrappers import FrameStack

SAVE_PATH = "./models"
GAMES = {"space_invaders": "ALE/SpaceInvaders-v5", "breakout": "ALE/Breakout-v5"}
MAX_FRAMES = 20000
MAX_BUFFER = 30
NB_FRAME = 4

# Configuration paramaters for the whole setup
seed = 42

@dataclass
class Stats:
    episode: np.ndarray
    score: np.ndarray
    steps_survived: np.ndarray


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
    model.compile(keras.optimizers.Adam(learning_rate=0.001), keras.losses.Huber())
    return model


def save(model, name, version):
    model.save(f"{SAVE_PATH}/{name}_v{version}")

def load(name: str):
    return tf.keras.models.load_model(f"{SAVE_PATH}/{name}")

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

def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    y_size, x_size = img.shape
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
    current_frame_delta = np.maximum(
        frames[..., 3] - frames[..., :3].mean(axis=-1), 0.0
    )
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    return img

def merge_frames(frames, to_preprocess: bool = False):
    if to_preprocess:
        frames = np.array([preprocess(f) for f in frames])
    else:
        frames = np.array(frames)

    if len(frames.shape) == 4:
        return apply_delta(np.concatenate(tuple(frames), axis=2))
    elif len(frames.shape) == 5:
        return np.array([merge_frames(f) for f in frames])

def calc_reward_mean(hisory_buffer, center = 13, dist = 8):
    res = 0
    for i in range(center-dist, center+dist+1):
        res += hisory_buffer[i][2]
    return res/(2*dist+1)

def train(name: str, version: int = 2, render: bool = False, stacked_frames: int = 4,):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)
    env = gym.make(GAMES[name], obs_type="rgb", render_mode='human' if render else None)
    env = FrameStack(env, stacked_frames, lz4_compress=True)
    env.seed(seed)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    gamma = 0.9  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    exploration_frames = MAX_FRAMES / 10
    epsilon_decay = 0.9 / exploration_frames

    n_actions = int(env.action_space.n)
    

    history_buffer = []
    #frames_observation = deque(maxlen=NB_FRAME)
    nb_buffer = 0

    observation, info = env.reset(seed=seed)
    #frames_observation.append(preprocess(observation, 2))

    #space_shape = preprocess(observation, 2).shape
    model = make_model(
        merge_frames(observation, to_preprocess=True), n_actions, version
    )

    stats = Stats(np.array([0]), np.array([0]), np.array([0]))
    running_reward = 0
    steps_survived = 0

    """for k in range(NB_FRAME-1):
        action = np.random.randint(n_actions)
        next_observation, reward, terminated, truncated, info = env.step(action)
        #frames_observation.append(preprocess(next_observation, 2))
        running_reward += reward
        steps_survived += 1"""

    for i in tqdm(range(MAX_FRAMES)):
        q_values = model(np.array([merge_frames(observation, to_preprocess=True)]))[0].numpy()
        e = np.random.rand()
        if e <= epsilon or nb_buffer<=MAX_BUFFER:
            # Select random action
            action = np.random.randint(n_actions)
        else:
            # Select action with highest q-value
            action = np.argmax(q_values)

        next_observation, reward, terminated, truncated, info = env.step(action)
        #frames_observation.append(preprocess(next_observation, 2))

        running_reward += reward
        steps_survived += 1


        # Append history_beffer with new observation
        history_buffer.append( (merge_frames(next_observation, to_preprocess=True), action, reward) )
        nb_buffer += 1

        if terminated:
            observation, info = env.reset(seed=seed)
            #frames_observation.append(preprocess(observation, 2))

            stats.episode = np.append(stats.episode, stats.episode[-1] + 1)
            stats.score = np.append(stats.score, running_reward)
            stats.steps_survived = np.append(stats.steps_survived, steps_survived)
            running_reward = 0
            steps_survived = 0

            continue

        #next_q_values = model(np.array([next_observation]))[0].numpy()
        # Maximum MAX_BEFFER elements in the buffer
        if nb_buffer>MAX_BUFFER:
            observation_temp, action_temp, _ = history_buffer.pop(0)
            #####
            next_q_values = model(np.array([merge_frames(observation, to_preprocess=True)]))[0].numpy()
            reward_temp = calc_reward_mean(history_buffer, 14, 9)
            nb_buffer -= 1
            update_q_values = q_values
            update_q_values[action_temp] = reward_temp + gamma * np.max(next_q_values)

            model.fit(np.array([observation_temp]), np.array([update_q_values]), verbose=0)

            if i<exploration_frames:
                epsilon -= epsilon_decay

        if not i%5000:
            save(model, name, version=version)
            save_stats(stats, name, version)
            print("model saved")

    save(model, name, version=version)
    save_stats(stats, name, version)

def test(name: str, version: int):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    env = gym.make(GAMES[name], obs_type="rgb", render_mode='human')
    env = FrameStack(env, 4, lz4_compress=True)
    env.seed(seed)
    model = load(f"{name}_v{version}")
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    #frames_observation = deque(maxlen=NB_FRAME)

    n_actions = int(env.action_space.n)

    while True:
        observation, _ = env.reset()
        #frames_observation.append(preprocess(observation, 2))
        terminated = False
        while not terminated:
          
            action = np.argmax(model(np.array([merge_frames(observation, to_preprocess=True)])))

            observation, reward, terminated, truncated, info = env.step(action)
            #frames_observation.append(preprocess(observation, 2))


if __name__ == "__main__":
    test("space_invaders", 2)
    #train("space_invaders", 2)
    #plot_stats("space_invaders", 2)