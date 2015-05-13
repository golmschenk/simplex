"""
The class for solving simplex.
Given a standard LPP, finds the optimal solution through simplex method.
"""

import numpy as np
from variable import Variable

class Simplex:
    """Class to preform simplex."""
    def __init__(self,
                 coefficients=np.array([[]], dtype='float'),
                 constraints=np.array([[]], dtype='float'),
                 objective=np.array([], dtype='float')):
        self.coefficients = coefficients
        self.constraints = constraints
        self.basis_objective = np.array([[]], dtype='float')
        self.basis_solution = np.array([[]], dtype='float')
        self.basis_value = 0
        self.value = 0
        self.solution = np.array([[]], dtype='float')
        self.reduced_costs = np.array([[]], dtype='float')
        self.least_positive_ratio = np.array([[]], dtype='float')
        self.basis_variables = []
        self.objective = objective
        self.basis_size = 0
        self.pivot_column_index = None
        self.pivot_row_index = None

    def initialize_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        basis_size = self.coefficients.shape[0]
        self.coefficients = np.append(self.coefficients, np.identity(basis_size), axis=1)
        self.objective = np.append(self.objective, np.zeros((basis_size), dtype='float'))

    def initialize_basis(self):
        """Sets up the initial basis."""
        self.basis_size = self.constraints.shape[0]
        self.basis_solution = self.constraints
        self.basis_objective = np.zeros(self.constraints.shape, dtype='float')
        self.basis_value = 0
        self.basis_variables = []
        for index in range(self.constraints.shape[0]):
            self.basis_variables.append(Variable(index=index, is_slack=True))

    def initialize_tableau(self):
        """Sets up the initial tableau values."""
        self.initialize_slack()
        self.initialize_basis()

    def calculate_basis_value(self):
        """Calculates the value for the basis objective function with the current solution."""
        self.basis_value = np.sum(np.inner(self.basis_objective.T, self.basis_solution.T))

    def calculate_reduced_costs(self):
        """Calculate the reduced costs of the current tableau."""
        temporary_reduced_costs = np.array([])
        for index, coefficient_column in enumerate(self.coefficients.T):
            product = np.inner(self.basis_objective.T, coefficient_column)
            temporary_reduced_costs = np.append(temporary_reduced_costs, product - self.objective[index])
        self.reduced_costs = temporary_reduced_costs

    def check_if_optimal(self):
        """Checks if the solution is optimal."""
        for reduced_cost in self.reduced_costs:
            if reduced_cost < 0:
                return False
        return True

    def check_if_unbounded(self):
        """Checks if the solution is unbounded."""
        for index, reduced_cost in enumerate(self.reduced_costs):
            if reduced_cost < 0:
                column = self.coefficients.T[index]
                if all(value <= 0 for value in column):
                    return True
        return False

    def obtain_pivot_column_index(self):
        """Return the column on which to pivot."""
        self.pivot_column_index =  np.argmin(self.reduced_costs)

    def obtain_pivot_row_index(self):
        """Return the row on which to pivot."""
        pivot_column = self.coefficients.T[self.pivot_column_index]
        self.least_positive_ratio = np.divide(self.basis_solution.flatten(), pivot_column)
        self.pivot_row_index = min([ratio for ratio in self.least_positive_ratio if ratio > 0])

    def make_pivot_element_one(self):
        """Multiply the pivot row to make the pivot element equal 1."""
        multiplier = 1.0 / self.coefficients[self.pivot_row_index][self.pivot_column_index]
        self.coefficients[self.pivot_row_index] *= multiplier
        self.basis_solution[self.pivot_row_index] *= multiplier
        self.basis_objective[self.pivot_row_index] *= multiplier

    def make_pivot_independent(self):
        """Preforms the row operations to make the pivot row independent."""
        for row_index in range(self.coefficients.shape[0]):
            if row_index != self.pivot_row_index:
                row_multiplier = self.coefficients[row_index][self.pivot_column_index]
                self.coefficients[row_index] -= row_multiplier * self.coefficients[self.pivot_row_index]
                self.basis_solution[row_index] -= row_multiplier * self.basis_solution[self.pivot_row_index]

    def swap_basis_variable(self):
        """Moves a new variable into the basis."""
        self.basis_objective[self.pivot_row_index][0] = self.objective[self.pivot_column_index]
        variable = Variable()
        if self.pivot_column_index >= self.basis_size:
            variable.is_slack = True
            variable.number = self.pivot_column_index - self.basis_size
        else:
            variable.is_slack = False
            variable.number = self.pivot_column_index
        self.basis_variables[self.pivot_row_index] = variable


