"""
The class for solving simplex.
Given a standard LPP, finds the optimal solution through simplex method.
"""

import numpy as np
from variable import Variable

class Simplex:
    """Class to preform simplex."""
    def __init__(self):
        self.coefficients = np.array([[]])
        self.constraints = np.array([[]])
        self.basis_objective = np.array([[]])
        self.basis_solution = np.array([[]])
        self.basis_value = 0
        self.value = 0
        self.solution = np.array([[]])
        self.reduced_costs = np.array([[]])
        self.least_positive_ratio = np.array([[]])
        self.basis_variables = []
        self.objective = np.array([[]])
        self.basis_size = 0

    def initialize_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        basis_size = self.coefficients.shape[0]
        self.coefficients = np.append(self.coefficients, np.identity(basis_size), axis=1)
        self.objective = np.append(self.objective, np.zeros((basis_size)))

    def initialize_basis(self):
        """Sets up the initial basis."""
        self.basis_size = self.constraints.shape[0]
        self.basis_solution = self.constraints
        self.basis_objective = np.zeros(self.constraints.shape)
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
        for reduced_cost in self.reduced_costs:
            if reduced_cost < 0:
                return False
        return True


