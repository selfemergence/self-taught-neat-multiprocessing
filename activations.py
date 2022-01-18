import logging

import torch
import torch.nn.functional as F

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return F.relu(x)
    
def softmax(x):
    return F.softmax(x, dim=1)


class Activations:
    
    def __init__(self):
        self.functions = dict(
                    sigmoid=sigmoid,
                    tanh=tanh,
                    relu=relu,
                    softmax=softmax
                )
    def get(self, func_name):
        return self.functions.get(func_name, None)
    
