# GA Genetic Algorithm for Analytic Functions

Genetic algorithm implemented in Python.
Toshiki Nazikian

## Description
The purpose of this project is to determine an analytic function from a dataset. The script generates a number of candidate functions and uses weighted random sampling to generate a new population. The weights are determined by the fitness scores for each candidate function based on mean square error. 

## Usage
Currently the script only works on pkl files storing numpy datasets such that the last column contains the target value. 
'''python
python ga.py 100 100 test  # prints out best candidate after 100 cycles
'''