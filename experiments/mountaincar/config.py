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

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    
    LOSS_FUNCTION = torch.nn.MSELoss()
    
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
    NUMBER_OF_GENERATIONS = 101
    SPECIATION_THRESHOLD = 3.0
    ELITISM = POPULATION_SIZE // 20
    
    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.1 #0.03, 0.1
    ADD_CONNECTION_MUTATION_RATE = 0.5
    
    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80
    

    def compute_training_fitness(self, index, render_test=False, queue=None):
        # OpenAI Gym
        env = copy.deepcopy(self.env)
        
        genome = self.population[index]
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
                
                inputs = torch.Tensor([observation]).to(self.DEVICE)
                outputs = phenotype(inputs)
                preds = outputs.cpu().clone().detach().numpy()[0][:self.NUM_OUTPUTS]
                action = int(np.argmax(preds))
                observation, reward, done, info = env.step(action)
                
                total_reward += reward
                
                # self teaching
                if not render_test:
                    action_outputs = outputs[0]#[:self.NUM_OUTPUTS]
                    teaching_outputs = outputs[0][-self.NUM_OUTPUTS:].detach()
                    teaching_outputs = torch.cat((teaching_outputs, teaching_outputs), dim=0)
                    phenotype.self_teaching(action_outputs, teaching_outputs)
                
                if done:
                    if render_test:
                        print("\tTotal reward ", total_reward)
                
            score.append(total_reward)
            average_reward = np.average(score)
            
        env.close()
            
        # put reward into queue
        if queue is not None:
            queue.put([average_reward, index])
            
        return [average_reward, index]
    
    def compute_test_reward(self, genome, render_test=True):
        env = copy.deepcopy(self.env)
        phenotype = FeedForwardNet(genome, self)
        
        score = []
        for episode in range(self.TEST_EPISODES):
            print("***Testing Episode ", episode)
            
            observation  = env.reset()
            done = False
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
                total_reward += reward
                
                # self teaching
                if not render_test:
                    outputs = pred[0][:self.NUM_OUTPUTS]
                    teaching_outputs = pred[0][-self.NUM_OUTPUTS:].detach()
                    phenotype.self_teaching(outputs, teaching_outputs)
                    
                if done:
                    if render_test:
                        print("\tReward: ", total_reward)
                
            score.append(total_reward)
            average_reward = np.average(score)
            print(f"\tAverage Reward after {episode} episode(s): {average_reward}")
                
        env.close()
            
        return average_reward