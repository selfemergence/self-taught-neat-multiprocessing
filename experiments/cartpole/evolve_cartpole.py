# relative import from another directory
import os
import sys
p = os.path.abspath('../..')
sys.path.insert(1, p)
import population as pop
import experiments.cartpole.config as c
from visualize import draw_net
#from tqdm import tqdm
# setting Seeds
import torch
torch.manual_seed(c.Config.SEED)
import numpy as np
np.random.seed(c.Config.SEED)
import random
random.seed(c.Config.SEED)

#from feed_forward import FeedForwardNet

#import torch.nn as nn


neat = pop.Population(c.Config)
bests = neat.run()

solutions = sorted(bests, key=lambda x: (x[0].fitness, -1*x[1]), reverse=True)
solution = solutions[0][0]

draw_net(solution, view=True, filename='./results/'+ c.Config.ENV_NAME +'-solution' , show_disabled=True)
        
# save solution
import pickle
# save best weights for future uses
with open("./results/" + c.Config.ENV_NAME +"-bests.plt","wb") as f:
    pickle.dump(bests, f)

with open("./results/" + c.Config.ENV_NAME +"-solution.plt","wb") as f:
    pickle.dump(solution, f)

