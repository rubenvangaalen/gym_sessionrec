from enum import Enum

import gym
import numpy as np
from gym import spaces


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


class MusicSessionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # required function = DRAFT IS DONE
    def __init__(self, df, window_size):
        assert df.ndim == 2  # Number of dimensions in the dataset (two-dimensional: rows x columns)

        self.seed()
        self.df = df  # Initialize dataframe
        self.window_size = window_size  # Initialize window_size

        self.signal_features = self._process_data()  # Get signal features out of database
        # 1 window observations, 3 columns (topic, skip, previous)
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))  # 16 options based on the topics
        # define observation space, 16 genres and boolean for skip: 16*2 = 32
        self.observation_space = spaces.Box(low=0, high=31, shape=self.shape, dtype=np.int8)

        # episode
        self._start_tick = self.window_size  # Start at 1
        self._end_tick = len(self.signal_features) - 1  # end 1 prior to last song, not sure if needed
        self._done = None  ## Initialize variable
        self._current_tick = None  ## Initialize variable
        self._total_reward = None  ## Initialize variable

    # CUSTOM function
    def _get_observation(self):
        # Create state matrix to consider before taking action
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    # required function
    def reset(self):
        self._done = False
        self._current_tick = self._start_tick

        self._total_reward = 0.

        return self._get_observation()  ##Start over again

    # required function
    def step(self, action):
        ##START: keep track of episode
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
        ##END: keep track of episode

        # return observation for current tick from which states are red
        observation = self._get_observation()

        step_reward = self._calculate_reward(action, observation)
        self._total_reward += step_reward

        # some info to print per tick
        info = dict(total_reward=self._total_reward)

        return observation, step_reward, self._done, info

    def render_all(self, mode='human'):
        # TODO render function
        print(self)



    ##CUSTOM function
    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError
