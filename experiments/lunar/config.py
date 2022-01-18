import torch
import gym
import numpy as np
import time
import copy

import sys
import os
p = os.path.abspath('../..')
sys.path.insert(1, p)

from feed_forward import FeedForwardNet

env_dict = gym.envs.registration.registry.env_specs.copy()

#env_name = 'LunarLander-v2'
#env1 = gym.make(env_name)
#state_size = env1.reset().shape[0]
#action_size = env1.action_space.n

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    
    LOSS_FUNCTION = torch.nn.BCELoss()
    
    SEED = 2021
    ENV_NAME = 'LunarLander-v2'
    env = gym.make(ENV_NAME)
    env.seed(SEED)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    reward_threshold = env.spec.reward_threshold
    
    NUM_INPUTS = state_size
    NUM_OUTPUTS = action_size
    USE_BIAS = True
    
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'softmax'
    SCALE_ACTIVATION = 4.9
    
    FITNESS_THRESHOLD = reward_threshold
    TRAIN_EPISODES = 1
    TEST_EPISODES = 100
    
    PROCESSES = 3
    
    POPULATION_SIZE = 200
    NUMBER_OF_GENERATIONS = 51
    SPECIATION_THRESHOLD = 3.0
    ELITISM = POPULATION_SIZE // 20
    
    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.3
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
        
        for episode in range(episodes):
            done = False
            observation  = env.reset()
            
            if render_test:
                print(">Testing Episode ", episode)
            
            total_reward = 0
            
            while not done:
                if render_test:
                    env.render()
                    time.sleep(0.005)
                
                input = torch.Tensor([observation]).to(self.DEVICE)
                pred = phenotype(input)
                actions = pred.clone().detach().numpy()[0][:self.NUM_OUTPUTS]
                action = int(np.argmax(actions))
                
                # self teaching
                if not render_test:
                    outputs = pred[0]#[:self.NUM_OUTPUTS]
                    teaching_outputs = pred[0][-self.NUM_OUTPUTS:].detach()
                    teaching_outputs = torch.cat((teaching_outputs, teaching_outputs), dim=0)
                    phenotype.self_teaching(outputs, teaching_outputs)
                    
                observation, reward, done, info = env.step(action)
                
                
                total_reward += reward
                
                if done:
                    if render_test:
                        print("\tTotal reward ", total_reward)
                
            score.append(total_reward)
            average_reward = np.average(score)
        
        env.close()
            
        return average_reward
    
    
    
            
        
        