"""Simplex examples."""
import numpy as np
from simplex import Simplex
from display import Display

def example1():
    coefficients = np.array([[1,  1],
                             [1, -1]], dtype='float')
    constraints = np.array([[4],
                            [2]], dtype='float')
    objective = np.array([3, 2])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display(simplex_init=simplex)
    display.run_simplex()

def example2():
    coefficients = np.array([[4.1, 2],
                             [2, 4]], dtype='float')
    constraints = np.array([[40],
                            [32]], dtype='float')
    objective = np.array([80, 55])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display(simplex_init=simplex)
    display.run_simplex()

def example3():
    coefficients = np.array([[2, 1,  0],
                             [1, 2, -2],
                             [0, 1,  2]], dtype='float')
    constraints = np.array([[10],
                            [20],
                            [ 5]], dtype='float')
    objective = np.array([2, -1, 2])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display(simplex_init=simplex)
    display.run_simplex()

def example4():
    coefficients = np.array([[1, -1],
                             [2, -1]], dtype='float')
    constraints = np.array([[10],
                            [40]], dtype='float')
    objective = np.array([2, 1])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display(simplex_init=simplex)
    display.run_simplex()
