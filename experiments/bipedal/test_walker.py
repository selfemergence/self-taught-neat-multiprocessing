import logging

import sys
import os
p = os.path.abspath('../..')
sys.path.insert(1, p)
import population as pop
import experiments.bipedal.config as c
from visualize import draw_net
from tqdm import tqdm

print("Start testing")

#neat = pop.Population(c.Config)
#solution, generation = neat.run()
        
# save solution
import pickle
# save best weights for future uses
file = open("./results/evolve-walker.plt",'rb')
best_genome = pickle.load(file)
file.close()

config = c.Config
fitness = config.fitness_fn(config, genome=best_genome, render_test=True)
print("test fitness ", fitness)


