"""Tests for the simplex module."""
import numpy as np
from simplex import Simplex
from variable import Variable


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

    def test_adding_slack_extends_objective_with_zeros(self):
        """Checks that the slack variables can be added for larger matrices."""
        simplex = Simplex()
        coefficients = np.array([[1,  1],
                                 [1, -1]])
        simplex.coefficients = coefficients
        simplex.objective = np.array([-3, -2])
        expected_objective = np.array([-3, -2, 0, 0])

        simplex.initialize_slack()

        assert np.array_equal(simplex.objective, expected_objective)

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
        assert np.array_equal(simplex.basis_objective, np.array([[0], [0]]))

        simplex = Simplex()
        simplex.constraints = np.array([[4],
                                        [2],
                                        [1]])
        simplex.initialize_basis()
        assert np.array_equal(simplex.basis_objective, np.array([[0], [0], [0]]))

    def test_initializing_basis_marks_slack_variables_as_basis(self):
        simplex = Simplex()
        simplex.constraints = np.array([[4],
                                        [2],
                                        [1]])

        simplex.initialize_basis()

        assert Variable(index=2, is_slack=True) in simplex.basis_variables

    def test_calculation_of_basis_value(self):
        simplex = Simplex()
        simplex.basis_objective = np.array([[1],
                                            [2]])
        simplex.basis_solution = np.array([[4],
                                           [2]])

        simplex.calculate_basis_value()

        assert simplex.basis_value == 8

    def test_can_make_tableau_from_constraints_coefficients_and_objective(self):
        simplex = Simplex()
        simplex.coefficients = np.array([[1,  1],
                                         [1, -1]])
        simplex.constraints = np.array([[4],
                                        [2]])

        simplex.initialize_tableau()

        expected_coefficients = np.array([[1,  1, 1, 0],
                                          [1, -1, 0, 1]])
        expected_basis_variables = [Variable(index=0, is_slack=True), Variable(index=1, is_slack=True)]
        expected_basis_objective = np.array([[0],
                                             [0]])
        expected_basis_solution = np.array([[4],
                                            [2]])
        np.array_equal(simplex.coefficients, expected_coefficients)
        assert simplex.basis_variables == expected_basis_variables
        np.array_equal(simplex.basis_objective, expected_basis_objective)
        np.array_equal(simplex.basis_solution, expected_basis_solution)

    def test_calculate_reduced_costs(self):
        simplex = Simplex()
        simplex.basis_size = 2
        simplex.coefficients = np.array([[1,  1, 1, 0],
                                         [1, -1, 0, 1]])
        simplex.basis_objective = np.array([[0],
                                            [0]])
        simplex.basis_solution = np.array([[4],
                                           [2]])
        simplex.objective = np.array([3, 2, 0, 0])

        simplex.calculate_reduced_costs()

        expected_reduced_costs = np.array([-3, -2, 0, 0])
        assert np.array_equal(simplex.reduced_costs, expected_reduced_costs)

    def test_checking_for_optimality(self):
        simplex = Simplex()
        simplex.reduced_costs = np.array([3, 0, 2])
        is_optimal = simplex.check_if_optimal()
        assert is_optimal

        simplex = Simplex()
        simplex.reduced_costs = np.array([3, 0, -1])
        is_optimal = simplex.check_if_optimal()
        assert not is_optimal

    def test_checking_for_unboundedness(self):
        simplex = Simplex()
        simplex.reduced_costs = np.array([3, 0, -1])
        simplex.coefficients = np.array([[1, 2, -1],
                                         [2, 1,  1]])
        is_unbounded = simplex.check_if_unbounded()
        assert not is_unbounded

        simplex = Simplex()
        simplex.reduced_costs = np.array([3, 0, -1])
        simplex.coefficients = np.array([[1, 2, -1],
                                         [2, 1,  0]])
        is_unbounded = simplex.check_if_unbounded()
        assert is_unbounded

    def test_pivot_column_attaining(self):
        simplex = Simplex()
        simplex.reduced_costs = np.array([3, -2, -1])

        simplex.obtain_pivot_column_index()

        assert simplex.pivot_column_index == 1

    def test_pivot_row_attaining(self):
        simplex = Simplex()
        simplex.pivot_column_index = 1
        simplex.coefficients = np.array([[1, 2, -1],
                                         [2, 1,  1],
                                         [2, 1,  1]])
        simplex.basis_solution = np.array([[-5], [1], [2]])

        simplex.obtain_pivot_row_index()

        assert simplex.pivot_row_index == 1