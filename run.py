import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register

from datasets import MSSD_Mini_Topics_Simple

# X should be random, but session_position column should be 1.
# Y should be all rows after X untill it starts at 1 again.
x = 4
y = 14

register('songs-v0',
         entry_point='envs.discrete_topics:Simple',
         kwargs={
             'df': MSSD_Mini_Topics_Simple,
             'window_size': 1,  # Look only at previous row for the state
             'frame_bound': (x, y)
         })

env = gym.make('songs-v0',
               df=MSSD_Mini_Topics_Simple,
               window_size=1,  # Look only at previous row for the state
               frame_bound=(x, y)
               )

observation = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()
