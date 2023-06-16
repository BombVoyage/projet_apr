import signal
import sys
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

SAVE_PATH = "./models"

GAMES = {"space_invaders": "ALE/SpaceInvaders-v5"}

# Configuration paramaters for the whole setup
seed = 42


def model_v0(input_shape, n_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(16, 9, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(16, 5, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(16, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(n_actions, activation="linear")(layer5)
    model = keras.Model(inputs=inputs, outputs=action)

    model.compile(
        keras.optimizers.Adam(learning_rate=0.01), keras.losses.MeanSquaredError()
    )
    return model


def save(model, name):
    model.save(f'{SAVE_PATH}/{name}')


def load(name: str):
    return tf.keras.models.load_model(f"{SAVE_PATH}/{name}")


def train(name: str):
    env = gym.make(GAMES[name], obs_type="rgb")
    env.seed(seed)

    gamma = 0.9  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_decay = 0.9
    max_steps_per_episode = 1000
    max_episodes = 50

    n_actions = int(env.action_space.n)
    space_shape = env.observation_space.shape
    model = model_v0(space_shape, n_actions)

    for _ in tqdm(range(max_episodes)):
        observation, info = env.reset(seed=seed)
        for _ in range(max_steps_per_episode):
            q_values = model(np.array([observation]))[0].numpy()

            e = np.random.rand()
            if e <= epsilon:
                # Select random action
                action = np.random.randint(n_actions)
            else:
                # Select action with highest q-value
                action = np.argmax(q_values)

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_q_values = model(np.array([next_observation]))[0].numpy()

            update_q_values = q_values
            update_q_values[action] = 1 / (1 + reward) + gamma * np.max(next_q_values)

            model.fit(np.array([observation]), np.array([update_q_values]))

            if terminated:
                break
        epsilon *= epsilon_decay
    save(model, name)


def test(name: str):
    def quit(env, sig, frame):
        env.close()
        sys.exit(0)

    env = gym.make(GAMES[name], obs_type="rgb", render_mode="human")
    env.seed(seed)
    model = load(name)
    signal.signal(signal.SIGINT, lambda sig, frame: quit(env, sig, frame))
    while True:
        observation, _ = env.reset()
        terminated = False
        while not terminated:
            action = np.argmax(model(np.array([observation])))
            observation, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":

    test("space_invaders")
