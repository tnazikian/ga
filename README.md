# GA Genetic Algorithm for Analytic Functions

Genetic algorithm implemented in Python.
Toshiki Nazikian

## Description
The purpose of this project is to determine an analytic function from a dataset. The script generates a number of candidate functions and uses weighted random sampling to generate a new population. The weights are determined by the fitness scores for each candidate function based on mean square error. 

## Usage
Currently the script only works on pkl files storing numpy datasets such that the last column contains the target value.
```python
python ga.py 1000 200 test  # prints out best candidate after 200 cycles
```
The two numbers represent the number of individuals to be generated, and the number of cycles, respectively. This particular example should model y = 10(x0)^3 + 4(x1)^2.

The coefficients and powers of the candidate functions are randomly selected in a default range of [-10, 10] for coefficients, and [-1, 5] for powers. The user can also specify custom ranges as well as what operators can be used in the functions:

```python
python ga.py 1000 200 test -cmin -5 -cmax 10 -emin -5 -emax 5 -operators +,-,*
```