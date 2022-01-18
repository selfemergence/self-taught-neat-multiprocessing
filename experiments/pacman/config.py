import torch
import gym
import numpy as np
import time

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
    
    NUM_INPUTS = 128
    NUM_OUTPUTS = 9
    USE_BIAS = True
    
    LOSS_FUNCTION = torch.nn.MSELoss()
    
    ACTIVATION = 'sigmoid'
    OUTPUT_ACTIVATION = 'relu'
    SCALE_ACTIVATION = 4.9
    
    FITNESS_THRESHOLD = 200
    TRAIN_EPISODES = 1
    TEST_EPISODES = 100
    
    PROCESSES=3
    
    SEED = 2021
    ENV_NAME = 'MsPacman-ram-v0' 
    
    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 301
    SPECIATION_THRESHOLD = 5.0
    ELITISM = 2
    
    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.1
    ADD_CONNECTION_MUTATION_RATE = 0.5
    
    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80
    
    def fitness_fn(self, genome, epochs=1, render_test=False):
        # OpenAI Gym
        env = gym.make(self.ENV_NAME)
        env.seed(self.SEED)
        
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
                observation, reward, done, info = env.step(action)
                
                # self teaching
                if not render_test:
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
        
        #print("Average Reward ", average_reward)
            
        return average_reward
    
    
    
            
        
        