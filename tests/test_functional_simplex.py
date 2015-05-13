"""Functional tests for the simplex module."""
import numpy as np
from simplex import Simplex
from variable import Variable


class TestFunctionalSimplex:
    """Functional tests for the simplex class."""
    def test_full_simplex(self):
        coefficients = np.array([[1,  1],
                                 [1, -1]])
        constraints = np.array([[4],
                                [2]])
        objective = np.array([3, 2])
        simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)

        simplex.run()

        assert simplex.value == 11
        assert np.array_equal(simplex.solution, np.array([3, 1]))