"""
Created on Sun Jul 18 17:56:02 2021

@author: namlh
"""
import time
import random
import numpy as np
import torch

from multiprocessing.pool import ThreadPool
from torch.multiprocessing import Pool
from torch import multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
import concurrent.futures


# relative import from another directory
import os
import sys
#p = os.path.abspath('../')
#sys.path.insert(1, p)

import utils
from genome import Genome
from species import Species
from crossover import crossover
from mutation import mutate

#logger = logging.getLogger(__name__)

class Population:
    __global_innovation_number = 0
    current_gen_innovation = []  # Can be reset after each generation according to paper

    def __init__(self, config):
        self.Config = config()
        if torch.cuda.is_available():
            config.DEVICE = "cuda"
            self.pool = ThreadPool(self.Config.PROCESSES)
        else:
            self.pool = Pool(self.Config.PROCESSES)
        
        
        self.fitness_function = self.Config.compute_training_fitness
        self.population = self.set_initial_population()
        self.species = []
        
        # statistics
        self.bests = []
        self.best_fitness = []
        self.average_fitness = []
        self.max_average_reward = []
        self.worst_fitness = []  

        for genome in self.population:
            self.speciate(genome, 0)

    def run(self):
        
        for generation in range(1, self.Config.NUMBER_OF_GENERATIONS):
            print("****Generation ", generation)
            self.Config.population = self.population
            # Get Fitness of Every Genome
            start = time.perf_counter()
            
            # using concurrent.features
            #fitnesses = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(self.fitness_function, index) for index in range(len(self.population))]
                #results = executor.map(self.fitness_function, range(len(self.population)))
                #print("****RESULTS", results, type(results), dir(results))
                for f in concurrent.futures.as_completed(results):
                    fitness = f.result()
                    index = fitness[1]
                    self.population[index].fitness = fitness[0]
####     
#            for f in fitnesses:
#                index = f[1]
#                self.population[index].fitness = f[0]
            
#           #using pool
#            fitnesses = self.pool.map(self.fitness_function, [index for index in range(len(self.population))])
#            for i in range(len(self.population)):
#                self.population[i].fitness = fitnesses[i]
#            
            # Normal serial fitness calculation
#            for i in  range(len(self.population)):
#                self.population[i].fitness = self.fitness_function(i)
            
            
            # using queue and process
#            q = multiprocessing.Queue()
#            processes = []
#            for i in range(len(self.population)):
#                p = multiprocessing.Process(target=self.fitness_function, 
#                                            args=[i, False, q])
#                p.start()
#                processes.append(p)
#            
#            fitnesses = []
#            for p in processes:
#                p.join()
#                fitnesses.append(q.get())
#                p.close()
#            q.close()
#            for f in fitnesses:
#                #print("Fitness and Index ", f)
#                index = f[1]
#                #fitness = f[0]
#                self.population[index].fitness = f[0]
                #print(index, fitness)
                   
            finish = time.perf_counter()
            print(f'\tFitness Calculation for generation {generation} Finished in {round(finish-start, 2)} second(s)')
                
            best_genome = utils.get_best_genome(self.population)
            # stats
            self.bests.append([best_genome, best_genome.connection_genes])
            self.best_fitness.append(best_genome.fitness)
            self.average_fitness.append(np.average([g.fitness for g in self.population]))
            

            # Reproduce
            all_fitnesses = []
            remaining_species = []

            for species, is_stagnant in Species.stagnation(self.species, generation):
                if is_stagnant:
                    self.species.remove(species)
                else:
                    all_fitnesses.extend(g.fitness for g in species.members)
                    remaining_species.append(species)

            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)

            fit_range = max(1.0, (max_fitness-min_fitness))
            for species in remaining_species:
                # Set adjusted fitness
                avg_species_fitness = np.mean([g.fitness for g in species.members])
                species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

            adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
            adj_fitness_sum = sum(adj_fitnesses)
            
            # culling species
            """loose bottom half of Species"""
            # Get the number of offspring for each species
            for species in remaining_species:
                if len(species.members) > 2:
                    species.members = species.members[:len(species.members)//2]
                    
            # reproduce new population
            new_population = []
            # Add elitism
            if self.Config.ELITISM > 0:
                self.population.sort(reverse=True)
                new_population.extend(self.population[:self.Config.ELITISM])
            
            # add members from remaining species
            sorted(remaining_species, key=lambda s: s.adjusted_fitness, reverse=True)
            best_species = remaining_species[0]
            best_members = best_species.members
            
            for species in remaining_species:    
                # compute number of children from a species
                if species.adjusted_fitness > 0:
                    noChildren = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.Config.POPULATION_SIZE))
                else:
                    noChildren = 2
                #print("\nNumber of offspring to be produced ", noChildren)
                
                # sort current members in order of descending fitness
                cur_members = species.members
                cur_members.sort(key=lambda g: g.fitness, reverse=True)
                species.members = []  # reset
                # save top individual in species
                new_population.append(cur_members[0])
                noChildren -= 1

                # Only allow top x% to reproduce
                purge_index = int(self.Config.PERCENTAGE_TO_SAVE * len(cur_members))
                purge_index = max(2, purge_index)
                cur_members = cur_members[:purge_index]

                for i in range(noChildren):
                    parent_1 = random.choice(cur_members)
                    parent_2 = random.choice(cur_members)

                    child = crossover(parent_1, parent_2, self.Config)
                    mutate(child, self.Config)
                    new_population.append(child)
            
            
            # fill in more individuals from best species  
            while len(new_population) < self.Config.POPULATION_SIZE: 
                parent_1 = random.choice(best_members)
                parent_2 = random.choice(best_members)
                child = crossover(parent_1, parent_2, self.Config)
                mutate(child, self.Config)
                new_population.append(child)
            
            # Set new population
            self.population = new_population
            Population.current_gen_innovation = []

            # Speciate
            for genome in self.population:
                self.speciate(genome, generation)

            #if best_genome.fitness >= self.Config.FITNESS_THRESHOLD:
                #print("FOUND A SOLUTION", best_genome.fitness)
                #return best_genome, generation

            # Generation Stats
            if self.Config.VERBOSE:
                print('\tFinished Generation', {generation})
                print('\tBest Genome Fitness:', {best_genome.fitness})
                print('\tBest Genome Length', len(best_genome.connection_genes))
                print('\tAverage Fitness ', round(self.average_fitness[-1], 2))
                #print("\tMIN ",self.worst_fitness[-1], " | AVG ", 
                #self.average_fitness[-1], " | MAX ", self.best_fitness[-1])
                #logger.info(f'Finished Generation {generation}')
                #logger.info(f'Best Genome Fitness: {best_genome.fitness}')
                #logger.info(f'Best Genome Length {len(best_genome.connection_genes)}\n')
            
        #self.bests = sorted(self.bests, key=lambda x: x.fitness, reverse=True)
        #self.bests = sorted(self.bests, key=lambda x: (x[0].fitness, x[1]), reverse=True)
        
        # save the results
        np.savetxt('results/' + self.Config.ENV_NAME + '.txt', list(zip(self.best_fitness, 
                                                            self.average_fitness)), \
               fmt='%.18g', delimiter='\t', header="Fitness: Best, Avg")

        import matplotlib.pyplot as plt
        
        plt.figure(1)
        plt.plot(self.best_fitness, 'r-')
        plt.plot(self.average_fitness, 'g:')
        plt.xlabel("Generation")
        #plt.xlim(xmin=0)
        #plt.ylim(ymin=0)
        plt.ylabel("Fitness")
        plt.legend(['Best', 'Avg'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0., ncol=3, mode="expand")
        plt.savefig('results/' + self.Config.ENV_NAME+'.png')
        
        
        #return best_genome
        return self.bests

    def speciate(self, genome, generation):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :param generation: Number of generation this speciation is occuring at
        :return: None
        """
        for species in self.species:
            if Species.species_distance(genome, species.model_genome) <= self.Config.SPECIATION_THRESHOLD:
                genome.species = species.id
                species.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), genome, generation)
        genome.species = new_species.id
        new_species.members.append(genome)
        self.species.append(new_species)

    def assign_new_model_genomes(self, species):
        species_pop = self.get_genomes_in_species(species.id)
        species.model_genome = random.choice(species_pop)

    def get_genomes_in_species(self, species_id):
        return [g for g in self.population if g.species == species_id]

    def set_initial_population(self):
        pop = []
        for i in range(self.Config.POPULATION_SIZE):
            new_genome = Genome()
            inputs = []
            outputs = []
            bias = None
            
            # Create nodes
            for j in range(self.Config.NUM_INPUTS):
                #print("\t Create input node ", j)
                n = new_genome.add_node_gene('input')
                inputs.append(n)

            # double outputs for self-teaching
            for j in range(2*self.Config.NUM_OUTPUTS):
                #print("\t Create output node ", j)
                n = new_genome.add_node_gene('output')
                outputs.append(n)

            if self.Config.USE_BIAS:
                #print("\t Create bias node ")
                bias = new_genome.add_node_gene('bias')
            
            
            # Create connections
            n = 0
            for input in inputs:
                for output in outputs:
                    n += 1
                    new_genome.add_connection_gene(input.id, output.id)

            if bias is not None:
                for output in outputs:
                    new_genome.add_connection_gene(bias.id, output.id)

            pop.append(new_genome)

        return pop

    @staticmethod
    def get_new_innovation_num():
        # Ensures that innovation numbers are being counted correctly
        # This should be the only way to get a new innovation numbers
        ret = Population.__global_innovation_number
        Population.__global_innovation_number += 1
        return ret
