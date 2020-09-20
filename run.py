import random
from typing import Tuple

import gym
import numpy as np
from gym.envs.registration import register

register('songs-v0',
         entry_point='envs.music_session_env:MusicSessionEnv',
         kwargs={'window_size': 2, 'session_length': 21})

env = gym.make('songs-v0', window_size=2, session_length=21)

# INITIALISE Q TABLE TO ZERO
Q = np.zeros((16, 2, 16, 2, 16))  # category[-2], skipped[-2], category[-1], skipped[-1], action

# HYPERPARAMETERS
train_episodes = 10000  # Total train episodes
test_episodes = 10  # Total test episodes
max_steps = 20  # Max steps per episode
alpha = 0.1  # Learning rate
gamma = 0.6  # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 0.01  # Exploration rate
max_epsilon = 1  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.001  # Exponential decay rate for exploration prob

# TRAINING PHASE
training_rewards = []  # list of rewards

# FOR GRAPH
simulations = 50
matrix = np.zeros((train_episodes, simulations))

for runs in range(simulations):
    for episode in range(train_episodes):
        state: Tuple[int, int, int, int] = env.reset()  # Reset the environment
        cumulative_training_rewards = 0

        for step in range(max_steps):
            # Choose an action (a) among the possible states (s)
            exp_exp_tradeoff = random.uniform(0, 1)  # choose a random number

            # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state])
            # Else choose a random action (Exploration)
            else:
                action = env.action_space.sample()

            # Perform the action (a) and observe the outcome state(s') and reward (r)
            observation, reward, done, info = env.step(action)

            # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Q[(*state, action)] = Q[(*state, action)] + alpha * (reward + gamma * np.max(Q[state]) - Q[(*state, action)])
            cumulative_training_rewards += reward  # increment the cumulative reward        
            state = observation  # Update the state

            # If we reach the end of the episode
            if done:
                #print("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # append the episode cumulative reward to the list
        training_rewards.append(cumulative_training_rewards)

    #print("Training score over time: " + str(sum(training_rewards) / train_episodes))
    matrix[:,runs] = training_rewards
    training_rewards = []
#print(matrix)

np.savetxt("RL_results.csv", matrix, delimiter=",")
