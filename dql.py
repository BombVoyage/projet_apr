import signal
import sys
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

SAVE_PATH = "./models"
GAMES = {"space_invaders": "ALE/SpaceInvaders-v5", "breakout": "ALE/Breakout-v5"}
MAX_FRAMES = 20000
MAX_BUFFER = 25

# Configuration paramaters for the whole setup
seed = 42


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
    model.compile(
        keras.optimizers.Adam(learning_rate=0.01), keras.losses.MeanSquaredError()
    )
    return model


def save(model, name, version):
    model.save(f"{SAVE_PATH}/{name}_v{version}")


def load(name: str):
    return tf.keras.models.load_model(f"{SAVE_PATH}/{name}")

def calc_reward_mean(hisory_buffer, center = 13, dist = 8):
    res = 0
    for i in range(center-dist, center+dist+1):
        res += hisory_buffer[i][2]
    return res/(2*dist+1)

def train(name: str, version: int = 2, render: bool = False):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)
    env = gym.make(GAMES[name], obs_type="rgb", render_mode='human' if render else None)
    env.seed(seed)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))

    gamma = 0.9  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    exploration_frames = MAX_FRAMES / 10
    epsilon_decay = 0.9 / exploration_frames

    n_actions = int(env.action_space.n)
    space_shape = env.observation_space.shape
    model = make_model(space_shape, n_actions, version)

    history_buffer = []
    nb_buffer = 0

    observation, info = env.reset(seed=seed)
    for _ in tqdm(range(MAX_FRAMES)):
        q_values = model(np.array([observation]))[0].numpy()

        e = np.random.rand()
        if e <= epsilon or nb_buffer<=MAX_BUFFER:
            # Select random action
            action = np.random.randint(n_actions)
        else:
            # Select action with highest q-value
            action = np.argmax(q_values)

        next_observation, reward, terminated, truncated, info = env.step(action)

        # Append history_beffer with new observation
        history_buffer.append( (observation, action, reward) )
        nb_buffer += 1

        if terminated:
            observation, info = env.reset(seed=seed)
            continue

        #next_q_values = model(np.array([next_observation]))[0].numpy()
        # Maximum MAX_BEFFER elements in the buffer
        if nb_buffer>MAX_BUFFER:
            observation_temp, action_temp, _ = history_buffer.pop(0)
            next_q_values = model(np.array([observation]))[0].numpy()
            reward_temp = calc_reward_mean(history_buffer, 13, 8)
            nb_buffer -= 1
            update_q_values = q_values
            update_q_values[action_temp] = reward_temp + gamma * np.max(next_q_values)

            model.fit(np.array([observation_temp]), np.array([update_q_values]))

            epsilon -= epsilon_decay
    save(model, name, version=version)

def test(name: str, version: int):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human")
    env.seed(seed)
    model = load(f"{name}_v{version}")
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))
    while True:
        observation, _ = env.reset()
        terminated = False
        while not terminated:
            action = np.argmax(model(np.array([observation])))
            observation, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    test("space_invaders", 2)
    #train("space_invaders")