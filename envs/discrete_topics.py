import numpy as np
import pandas as pd

from .session_env import MusicSessionEnv

###
# STEP 1:
# Retrieve matrix with rewards.
# Column = topic of action.
# Row = topic of previous song.
###

# Dataset with rows for topics not skipped previously
historic_data_no_skip = [list(row)[1:] for row in pd.read_csv("datasets/data/Reward_Previous_No_Skip.csv").values]

# Dataset with rows for topics skipped previously
historic_data_skip = [list(row)[1:] for row in pd.read_csv("datasets/data/Reward_Previous_Skip.csv").values]


class Simple(MusicSessionEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2  ##Throw an error if frame_bound does not have 2 items -> df[start:end]

        self.frame_bound = frame_bound  ##define
        super().__init__(df, window_size)  ##define

    # CUSTOM function
    def _process_data(self):

        ###
        ## CHANGE THE LINE BELOW FOR CONTINUES SCORES
        ### 
        topics = self.df.loc[:, 'topic'].to_numpy()  # Grab genre column
        not_skipped = self.df.loc[:, 'not_skipped'].to_numpy()  # Grab genre column
        session_position = self.df.loc[:, 'session_position'].to_numpy()  # Grab genre column

        topics[self.frame_bound[0]]  # NOT SURE; print the index of where to start in the dataframe as a check?

        topics = topics[self.frame_bound[0]:self.frame_bound[
            1]]  ##Select rows to consider for episode (max is 20 -> max(sessionLength))
        not_skipped = not_skipped[
                      self.frame_bound[0]:self.frame_bound[1]]  ##Get values of rows in episode for not skipped?
        session_position = session_position[self.frame_bound[0]:self.frame_bound[1]]

        signal_features = np.column_stack((topics, not_skipped, session_position))  # Create matrix of state columns

        return signal_features

    def _calculate_reward(self, action, observation):

        ###
        # STEP 2:
        # Calculate reward
        ###

        previous_topic = observation[0, 0]
        previous_skip = observation[0, 1]
        action_topic = action

        if previous_skip == 0:
            step_reward = historic_data_no_skip[previous_topic][action_topic]
        else:
            step_reward = historic_data_skip[previous_topic][action_topic]

        return step_reward
