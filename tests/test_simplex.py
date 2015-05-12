"""Tests for the simplex module."""
import numpy as np
from simplex import Simplex


class TestSimplex:
    """Tests for the simplex class."""
    def test_can_add_slack_variables_to_matrix(self):
        """Checks that the slack variables can be added correctly."""
        simplex = Simplex()
        coefficients = np.array([[1,  1],
                                 [1, -1]])
        simplex.coefficients = coefficients
        expected_coefficients = np.array([[1,  1, 1, 0],
                               [1, -1, 0, 1]])

        simplex.initialize_slack()

        assert np.array_equal(simplex.coefficients, expected_coefficients)

    def test_can_add_slack_variables_for_different_size_matrices(self):
        """Checks that the slack variables can be added for larger matrices."""
        simplex = Simplex()
        coefficients = np.array([[1,  1, 1],
                                 [1, -1, 2],
                                 [2,  1, 1]])
        simplex.coefficients = coefficients
        expected_a = np.array([[1,  1, 1, 1, 0, 0],
                               [1, -1, 2, 0, 1, 0],
                               [2,  1, 1, 0, 0, 1]])

        simplex.initialize_slack()

        assert np.array_equal(simplex.coefficients, expected_a)

    def test_initializing_basis_sets_the_basis_size(self):
        """Should set the basis based on the number of variables."""
        simplex = Simplex()
        coefficients = np.array([[1,  1],
                                 [1, -1]])
        simplex.coefficients = coefficients

        simplex.initialize_basis()

        assert simplex.basis_size == 2

    def test_can_make_tableau_from_constraints(self):
        # coefficients = np.array([[]])
        assert False # Finish me.