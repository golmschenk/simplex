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

example1()