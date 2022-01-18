from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import wrap_nes

import numpy as np

env_name = 'SuperMarioBros-v0'
env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

'''done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()'''

def prepro(I):
  """ prepro 240x256x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

state = env.reset()
new_state = prepro(state)