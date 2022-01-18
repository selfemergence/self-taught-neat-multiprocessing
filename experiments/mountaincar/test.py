import sys
import os
p = os.path.abspath('../..')
sys.path.insert(1, p)
import population as pop
import experiments.mountaincar.config as c
from visualize import draw_net
from tqdm import tqdm

import gym


neat = pop.Population(c.Config)
#solution, generation = neat.run()
        
# save solution
import pickle
# save best weights for future uses
file = open("./results/" + c.Config.ENV_NAME +"-bests.plt",'rb')
bests = pickle.load(file)
file.close()

bests = sorted(bests, key=lambda x: (x[0].fitness, -1*x[1]), reverse=True)
best_genome = bests[0][0]
print("***Expected Reward ", best_genome.fitness)

config = c.Config
reward = config.compute_test_reward(config, genome=best_genome, render_test=True)
print("test reward ", reward)


