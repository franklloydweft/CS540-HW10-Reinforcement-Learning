import pickle
import random
from collections import defaultdict
from collections import deque

import gym
import numpy as np

EPISODES = 20000
LEARNING_RATE = .1  # alpha
DISCOUNT_FACTOR = .99  # gamma
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        while not done:
            # epsilon greedy behavior policy
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                np_Q = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action = np.argmax(np_Q)

            next_obs, reward, done, info = env.step(action)

            prev = Q_table[(obs, action)]
            np_Q_new = np.array([Q_table[(next_obs, i)] for i in range(env.action_space.n)])
            next_max = np.max(np_Q_new)

            # update policy
            if not done:
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * prev + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
            elif done:
                Q_table[(obs, action)] = ((1 - LEARNING_RATE) * prev) + (LEARNING_RATE * reward)

            obs = next_obs
            episode_reward = episode_reward + reward

        EPSILON = EPSILON * EPSILON_DECAY

        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    model_file.close()
    #########################
