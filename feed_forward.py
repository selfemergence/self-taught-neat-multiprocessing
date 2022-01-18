"""
Created on Sun Jul 18 19:44:02 2021

@author: namlh
"""
import logging
import torch
import torch.nn as nn
from torch import autograd

# relative import from another directory
import os
import sys
#p = os.path.abspath('../utils')
#sys.path.insert(1, p)

import activations as a

#logger = logging.getLogger(__name__)

# TODO: Multiple GPU support get from config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedForwardNet(nn.Module):

    def __init__(self, genome, config):
        super(FeedForwardNet, self).__init__()
        self.genome = genome
        self.units = self.build_units()
        self.lin_modules = nn.ModuleList()
        self.config = config
        self.activation = a.Activations().get(config.ACTIVATION)
        self.output_activation = a.Activations().get(config.OUTPUT_ACTIVATION)

        for unit in self.units:
            self.lin_modules.append(unit.linear)
            
        self.loss_function =  self.config.LOSS_FUNCTION

    def forward(self, x):
        outputs = dict()
        input_units = [u for u in self.units if u.ref_node.type == 'input']
        output_units = [u for u in self.units if u.ref_node.type == 'output']
        #print(output_units)
        # double output units for teaching nodes
        #output_units.extend(output_units)
        bias_units = [u for u in self.units if u.ref_node.type == 'bias']
        stacked_units = self.genome.order_units(self.units)

        # Set input values
        for u in input_units:
            outputs[u.ref_node.id] = x[0][u.ref_node.id]

        # Set bias value
        for u in bias_units:
            outputs[u.ref_node.id] = torch.ones((1, 1)).to(device)[0][0]

        # Compute through directed topology
        while len(stacked_units) > 0:
            current_unit = stacked_units.pop()

            if current_unit.ref_node.type != 'input' and current_unit.ref_node.type != 'bias':
                # Build input vector to current node
                inputs_ids = self.genome.get_inputs_ids(current_unit.ref_node.id)
                in_vec = autograd.Variable(torch.zeros((1, len(inputs_ids)), device=device, requires_grad=True))

                for i, input_id in enumerate(inputs_ids):
                    in_vec[0][i] = outputs[input_id]

                # Compute output of current node
                linear_module = self.lin_modules[self.units.index(current_unit)]
                if linear_module is not None:  # TODO: Can this be avoided?
                    #scaled = self.config.SCALE_ACTIVATION * linear_module(in_vec)
                    scaled = linear_module(in_vec)
                    out = self.activation(scaled)
                else:
                    out = torch.zeros((1, 1))

                # Add to outputs dictionary
                outputs[current_unit.ref_node.id] = out

        # Build output vector
        output = autograd.Variable(torch.zeros((1, len(output_units)), device=device, requires_grad=True))
        for i, u in enumerate(output_units):
            output[0][i] = outputs[u.ref_node.id]
            
        # compute activation for outputs
        output = self.output_activation(output)
            
        #self.output = output
        return output
    
    # Self-teaching model
    def self_teaching(self, outputs, teaching_outputs):
        '''
        loss is defined as the difference between output and teaching nodes
        weights are updated by backpropagation
        '''
        # set the action brain in training mode
        self.train()
        
        # define optimizer
        adam_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        #set the gradients to zero before starting to do backpropragation 
        adam_optimizer.zero_grad()
        
        #Caculate loss
        loss = self.loss_function(outputs, teaching_outputs)
        #print("Loss = ", loss)

        #print(loss, loss.grad)
        # require auto-grad for loss only if output has been detached from graph
        #loss.requires_grad = True
        
        #Backpropogate
        loss.backward()
        
        #Update weights
        adam_optimizer.step()
        

    def build_units(self):
        units = []

        for n in self.genome.node_genes:
            in_genes = self.genome.get_connections_in(n.id)
            num_in = len(in_genes)
            weights = [g.weight for g in in_genes]

            new_unit = Unit(n, num_in)
            new_unit.set_weights(weights)

            units.append(new_unit)
        return units


class Unit:

    def __init__(self, ref_node, num_in_features):
        self.ref_node = ref_node
        self.linear = self.build_linear(num_in_features)

    def set_weights(self, weights):
        if self.ref_node.type != 'input' and self.ref_node.type != 'bias':
            weights = torch.cat(weights).unsqueeze(0)
            for p in self.linear.parameters():
                p.data = weights

    def build_linear(self, num_in_features):
        if self.ref_node.type == 'input' or self.ref_node.type == 'bias':
            return None
        return nn.Linear(num_in_features, 1, False)

    def __str__(self):
        return 'Reference Node: ' + str(self.ref_node) + '\n'
                
            