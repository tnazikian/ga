#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/7/19

Takes in a list of individuals and dataset and generates 
a new population of at least 100 candidate functions for 
each evolutionary cycle. Each cycle consists of weighted random
resampling with replacement based on fitness scores, and a 
chromosome swapping and mutation phase.

"""
import copy, random, argparse, pickle
import numpy as np
from tqdm import tqdm
from helper import *
from Individual import Individual

MATE=0.4            #proportion of total population chosen to mate
MUTATE=0.5          #proportion of offspring that mutate
INDIVIDUALS=100     #minimum number of individuals in a population
class Population:
    def __init__(self, individuals, data, y):
        self.individuals=individuals
        if len(individuals) < INDIVIDUALS:
            self.pop_size=INDIVIDUALS
        else:
            self.pop_size=len(individuals)
        self.data = data
        self.y = y
        # Hyperparams for mutation of coefficient and power terms
        self.mut_coeffs = np.asarray([0.1, 1.0])
        # Calculate fitness scores of initial population
        self.scores = [individual.calc_fitness(self.data, self.y) 
             for individual in individuals]
        self.best_ind = np.where(self.scores == np.max(self.scores))[0][0]
        # used to compare best score of previous population to new population
        self.last_best_func = self.individuals[self.best_ind]
        
    def get_best_func(self):
        # Return current best candidate function
        return self.individuals[self.best_ind]
        
    def new_pop(self, scores):
        # Random weighted resampling of current population based on fitness
        sum1 = np.sum(scores)
        prob=scores/sum1
        new_pop = np.random.choice(self.individuals, self.pop_size, p=prob, replace=True)
        return new_pop
        
    def cycle(self):
        new_pop = self.new_pop(self.scores)
        # creates deep copies of chosen individuals to make new population
        self.individuals = [individual.get_copy() for individual in new_pop]
        mate_ind = self.mate()
        if len(mate_ind) > 0:
            self.mutate(mate_ind)
        self.scores = [individual.calc_fitness(self.data, self.y) 
             for individual in self.individuals]
        current_best_ind = np.where(self.scores == np.max(self.scores))[0][0]
        worst_score_ind = np.where(self.scores == np.min(self.scores))[0][0]
        # If best score of current pop < previous pop, replace worst performing
        # candidate with previous best candidate
        if self.last_best_func.get_fitness() > self.scores[current_best_ind]:
            self.scores[worst_score_ind] = self.last_best_func.get_fitness()
            self.individuals[worst_score_ind] = self.last_best_func
            self.best_ind = worst_score_ind
        else:
            self.best_ind = current_best_ind
        self.last_best_func = self.individuals[self.best_ind]

    def mate(self):
        num_to_mate = int(len(self.individuals)*MATE)
        if num_to_mate >= 2:
            if num_to_mate % 2 == 1:
                num_to_mate += 1    # Makes sure there is even number of parents
            mating_ind = np.random.choice(len(self.individuals), num_to_mate, replace=False)
            for i in range(len(mating_ind)//2):
                self.individuals[2*i].mate(self.individuals[2*i+1])
            return mating_ind
        return None
                
    def mutate(self, mate_ind):
        num_to_mutate = int(len(mate_ind)*MUTATE)
        mutate_ind = np.random.choice(mate_ind, num_to_mutate, replace=False)
        if len(mutate_ind) > 0:
            for i in mutate_ind:
                self.individuals[i].mutate(self.mut_coeffs * 
                        sigmoid_gyaku(self.get_best_func().get_fitness()/100))
    
    @staticmethod
    def generate_individuals(n, num_vars, coeff_min, coeff_max, exp_min, exp_max, operators):
        """
        Static method for generating initial population of size n from 
        a dataset with num_vars variables. Coefficient and exponential values of 
        chromosomes are randomly generated within a user-specified range. operators argument 
        contains list of strings that represent operators that can be used e.g. ['+', '-', '*'].
        """
        indiv = []
        if num_vars > 1 and len(operators) == 0:
            raise ValueError("no operators")
        for i in range(n):
            l = [np.random.uniform(low=coeff_min, high=coeff_max) for _ in range(num_vars)]
            a = [np.random.randint(low=exp_min, high=exp_max) for _ in range(num_vars)]
            m = np.random.choice(operators, len(l)-1)
            indiv.append(Individual(l, a, m))
        return indiv


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_individuals', type=int)
    parser.add_argument('n_cycles', type=int)
    parser.add_argument('datapath')
    parser.add_argument("-cmin", default=-10)
    parser.add_argument("-cmax", default=10)
    parser.add_argument("-emin", default=-1)
    parser.add_argument("-emax", default=5)
    parser.add_argument("-operators", default=['+','-','*','/'])
    args = parser.parse_args()

    f = open(args.datapath, 'rb')
    data = pickle.load(f)
    n_vars = len(data[0]) - 1
    num_cycles = args.n_cycles
    num_individuals = args.n_individuals
    x = data[:, :-1]
    y = data[:, -1]
    f.close()

    # generate individuals
    individuals = Population.generate_individuals(num_individuals, n_vars, args.cmin, 
                        args.cmax, args.emin, args.emax, args.operators)

    d = Population(individuals, x, y)

    fitnesses = []      # track best fitness score out of population
    x = []        
    # Training      
    for i in tqdm(range(num_cycles)):
        d.cycle()
        best = d.get_best_func()
        fitnesses.append(best.get_fitness())
        #print(best)
        x.append(i)
    print(best)