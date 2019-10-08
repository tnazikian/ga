"""
Toshiki Nazikian 10/7/19

Represents a candidate polynomial function. 
Representation consists of three numpy arrays: coefficients, 
powers, and operators such that each variable in the dataset
is assigned one power and coefficient, and there is one operator 
between each term of the function.
"""
import copy, random
import numpy as np

class Individual:
    def __init__(self, coeffs, pows, operators):
        if len(coeffs) - 1 != len(operators) or len(coeffs) != len(pows):
            raise ValueError("check inputs")
        self.coeffs = coeffs
        self.pows = pows
        self.num_vars = len(coeffs)
        self.operators = operators
        self.fitness = None
        
    def mutate(self, coeffs):
        """Randomly tweaks either the coefficient or power of one variable"""
        r = random.random()
        if r <= 0.75:
            c1 = np.random.choice(len(self.coeffs), 1)[0]
            self.coeffs[c1] += coeffs[0]*np.random.choice([-1, 1])
        else:
            c2 = np.random.choice(len(self.pows), 1)[0]
            self.pows[c2] += np.random.choice([-1, 1])
        
    def mate(self, individual):
        """randomly swaps one chromosome (operator, coefficient and power of one variable)
         with another individual"""
        choice = np.random.randint(low=0, high=len(self.coeffs))
        c_switch = self.coeffs[choice]
        p_switch = self.pows[choice]
        if choice != 0:  
            o_switch = self.operators[choice-1]
        
        #switch coeffs
        self.coeffs[choice] = individual.coeffs[choice]
        self.pows[choice] = individual.pows[choice]
        if choice != 0:
            self.operators[choice-1] = individual.operators[choice-1]
        individual.coeffs[choice] = c_switch
        individual.pows[choice] = p_switch
        if choice != 0:
            individual.operators[choice-1] = o_switch
            
    def calc_fitness(self, data, y, error_func='mse'):
        """Uses mean square error to generate a fitness score"""
        f = 0
        for i in range(len(data)):
            if error_func=='mse':
                err = (y[i]-self.calc_row(data[i]))**2
            else:
                err = np.abs(y[i]-self.calc_row(data[i]))
            f += err
        mean_error = f/len(data)
        self.fitness = 1000*(1/(1+mean_error))
        return self.fitness
    
    def get_fitness(self):
        return self.fitness
    
    def set_fitness(self, f):
        self.fitness = f
    
    def calc_row(self, row):
        s = ""
        for i in range(self.num_vars):
            if row[i] == 0:
                row[i] = row[i] + 0.01
            if i != self.num_vars - 1:
                s += "{0}*({1})**{2} {3} ".format(self.coeffs[i], row[i], self.pows[i], self.operators[i])
            else:
                s += "{0}*({1})**{2} ".format(self.coeffs[i], row[i], self.pows[i])
        return eval(str(s))

    def get_coeffs(self):
        return self.coeffs, self.pows, self.operators
    
    def get_copy(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        s = ""
        for i in range(self.num_vars):
            if i != self.num_vars - 1:
                s += "{0}*(x{1})**{2} {3} ".format(self.coeffs[i], i, self.pows[i], self.operators[i])
            else:
                s += "{0}*(x{1})**{2} ".format(self.coeffs[i], i, self.pows[i])
        return s