import torch
import gym
import numpy as np
import time
import copy

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import wrap_nes

import sys
import os
p = os.path.abspath('../..')
sys.path.insert(1, p)

from feed_forward import FeedForwardNet

#env_dict = gym.envs.registration.registry.env_specs.copy()

def prepro(I):
  """ prepro 240x256x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  #I = I[::2,::2]
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

#env_name = 'SuperMarioBros-v0'
#env = gym_super_mario_bros.make(env_name)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
#state_size = prepro(env.reset()).shape[0]
#action_size = env.action_space.n

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    
    SEED = 2021
    ENV_NAME = 'SuperMarioBros-v0' 
    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    state_size = prepro(env.reset()).shape[0]
    action_size = env.action_space.n
    reward_threshold = env.spec.reward_threshold
    env.seed(SEED)
    
    
    NUM_INPUTS = state_size
    NUM_OUTPUTS = action_size
    USE_BIAS = True
    
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'relu'
    SCALE_ACTIVATION = 4.9
    
    FITNESS_THRESHOLD = env.spec.reward_threshold
    TRAIN_EPISODES = 1
    TEST_EPISODES = 100
    
    
    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 101
    SPECIATION_THRESHOLD = 3.0
    ELITISM = 0
    
    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5
    
    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80
    
    def fitness_fn(self, genome, epochs=1, render_test=False):
        
        # OpenAI Gym
        env = copy.deepcopy(self.env)
        
        phenotype = FeedForwardNet(genome, self)
        
        score = []
        if render_test:
            episodes = self.TEST_EPISODES
        else:
            episodes = self.TRAIN_EPISODES
        
        
        start = time.perf_counter()
        for episode in range(episodes):
            done = False
            observation  = env.reset()
            # preprocess the observation
            observation = prepro(observation)
            
            if render_test:
                print(">Testing Episode ", episode)
            
            total_reward = 0
            
            while not done:
                if render_test:
                    env.render()
                    time.sleep(0.005)
                
                input = torch.Tensor([observation]).to(self.DEVICE)
                pred = phenotype(input)
                actions = pred.clone().detach().numpy()[0]
                action = int(np.argmax(actions))
                observation, reward, done, info = env.step(action)
                
                # self teaching
                #if not render_test:
                outputs = pred[0]#[:self.NUM_OUTPUTS]
                teaching_outputs = pred[0][-self.NUM_OUTPUTS:].detach()
                teaching_outputs = torch.cat((teaching_outputs, teaching_outputs), dim=0)
                phenotype.self_teaching(outputs, teaching_outputs)
                
                total_reward += reward
                
                if done:
                    if render_test:
                        print("\tTotal reward ", total_reward)
                
            score.append(total_reward)
            average_reward = np.average(score)
        
        if render_test:
            env.close()
            
        finish = time.perf_counter()
        print(f'\tFitness calculation finished in {round(finish-start,2)} second(s)')
            
        return average_reward
    
    
    
            
        
        