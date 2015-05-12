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

    def test_initializing_basis_sets_the_basis_initial_solution_to_the_initial_constraints(self):
        """Should set the basis based on the number of variables."""
        simplex = Simplex()
        simplex.coefficients = np.array([[1,  1],
                                         [1, -1]])
        simplex.constraints = np.array([[4],
                                        [2]])
        expected_basis_solution = np.array([[4],
                                            [2]])

        simplex.initialize_basis()

        assert np.array_equal(simplex.basis_solution, expected_basis_solution)

    def test_initializing_basis_creates_a_basis_solution_of_zeros_of_the_right_size(self):
        simplex = Simplex()
        simplex.constraints = np.array([[4],
                                        [2]])
        simplex.initialize_basis()
        assert np.array_equal(simplex.basis_coefficients, np.array([[0], [0]]))

        simplex = Simplex()
        simplex.constraints = np.array([[4],
                                        [2],
                                        [1]])
        simplex.initialize_basis()
        assert np.array_equal(simplex.basis_coefficients, np.array([[0], [0], [0]]))

    def test_can_make_tableau_from_constraints(self):
        # coefficients = np.array([[]])
        assert False # Finish me.