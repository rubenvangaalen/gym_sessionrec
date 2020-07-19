import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register

register('songs-v0',
         entry_point='envs.music_session_env:MusicSessionEnv',
         kwargs={'window_size': 1, 'session_length': 20})

env = gym.make('songs-v0', window_size=1, session_length=20)

for i in range(20):  # For 20 episodes
    observation = env.reset()
    while True:
        # Take a random action. TODO: use Q-learning to actually learn between episodes.
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            print("info:", info)
            break

plt.cla()
env.render_all()
plt.show()
