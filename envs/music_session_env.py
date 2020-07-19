from enum import Enum
from random import random, randrange

import gym
import pandas as pd
from gym import spaces

###
# Retrieve matrix with rewards.
# Column = topic of action.
# Row = topic of previous song.
###

# Dataset with rows for topics not skipped previously
historic_data_no_skip = [list(row)[1:] for row in pd.read_csv("datasets/data/Reward_Previous_No_Skip.csv").values]

# Dataset with rows for topics skipped previously
historic_data_skip = [list(row)[1:] for row in pd.read_csv("datasets/data/Reward_Previous_Skip.csv").values]


class Actions(Enum):
    Indie = 0
    Christian = 1
    Reggae = 2
    House = 3
    NewRap = 4
    Rock = 5
    Spanish = 6
    Throwback = 7
    Metal = 8
    Chill = 9
    Hiphop = 10
    Study = 11
    Summer = 12
    Christmas = 13
    Country = 14
    OldRap = 15


NUM_CATEGORIES = len(Actions)

SKIP = 0
NO_SKIP = 1


class MusicSessionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # required function = DRAFT IS DONE
    def __init__(self, window_size, session_length):

        self.seed()
        self.window_size = window_size  # Initialize window_size

        # spaces
        self.action_space = spaces.Discrete(NUM_CATEGORIES)  # 16 options based on the topics
        # define observation space, 16 genres and boolean for skip: 16*2 = 32
        self.observation_space = spaces.Tuple((
            spaces.Discrete(NUM_CATEGORIES),  # Previous topic
            spaces.Discrete(2)))  # Whether previous song was skipped or not

        # episode
        self._start_state = None
        self._start_tick = self.window_size  # Start at 1
        self._end_tick = session_length  # Set end tick from constructor parameter
        self._done = None  # Initialize variable
        self._current_tick = None  # Initialize variable
        self._total_reward = None  # Initialize variable
        self._history = None  # Initialize variable

    # required function
    def reset(self):
        # start with a given music topic in a session
        # assumed is that this one has not been skipped
        self._start_state = (randrange(NUM_CATEGORIES), NO_SKIP)
        self._done = False
        self._current_tick = self._start_tick
        self._total_reward = 0.0
        self._history = [self._start_state]

        return self._start_state

    # required function
    def step(self, action):
        # START: keep track of episode
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
        # END: keep track of episode

        previous_topic, previous_skip = self._history[-1]  # Get last action and skip from history

        # Will return a 1 or 0.
        skip = self._calculate_skip(previous_topic, previous_skip, action)

        if skip == SKIP:
            next_topic = previous_topic
        else:
            next_topic = action

        observation = (next_topic, skip)
        self._history.append(observation)

        self._total_reward += skip

        # some info to print per tick
        info = dict(total_reward=self._total_reward)

        # return observation, reward, done, info
        return observation, skip, self._done, info

    def render_all(self, mode='human'):
        # TODO render function
        print(self)

    # CUSTOM functions

    def _process_data(self):
        raise NotImplementedError

    def _calculate_skip(self, previous_topic, previous_skip, action_topic):
        if previous_skip == 0:
            no_skip_probability = historic_data_no_skip[previous_topic][action_topic]
        else:
            no_skip_probability = historic_data_skip[previous_topic][action_topic]

        # Generate random number between 0 and 1
        random_number = random()

        # if random_number is < no_skip_probability return 1, else 0
        if random_number < no_skip_probability:
            return NO_SKIP
        else:
            return SKIP
