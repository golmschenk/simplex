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

    def initialize_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        basis_size = self.coefficients.shape[0]
        self.coefficients = np.append(self.coefficients, np.identity(basis_size), axis=1)

    def initialize_basis(self):
        """Sets up the initial basis."""
        self.basis_solution = self.constraints
        self.basis_objective = np.zeros(self.constraints.shape)
        self.basis_value = 0
        self.basis_variables = []
        for index in range(self.constraints.shape[0]):
            self.basis_variables.append(Variable(index=index, is_slack=True))

    def initialize_tableau(self):
        self.initialize_slack()
        self.initialize_basis()

    def calculate_basis_value(self):
        self.basis_value = np.sum(np.inner(self.basis_objective.T, self.basis_solution.T))

    def calculate_reduced_costs(self):
        pass

