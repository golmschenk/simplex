"""Functional tests for the simplex module."""
import numpy as np
from simplex import Simplex


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

        assert simplex.is_optimal
        assert simplex.value == 11
        assert np.array_equal(simplex.solution, np.array([[3], [1]]))

    def test_full_simplex2(self):
        coefficients = np.array([[ 1,  1],
                                 [ 3, -8],
                                 [10,  7]])
        constraints = np.array([[ 4],
                                [24],
                                [35]])
        objective = np.array([5, 7])
        simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)

        simplex.run()

        assert simplex.is_optimal
        assert simplex.value == 28
        assert np.array_equal(simplex.solution, np.array([[0], [4]]))