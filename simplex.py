"""
The class for solving simplex.
Given a standard LPP, finds the optimal solution through simplex method.
"""

import numpy as np

class Simplex:
    """Class to preform simplex."""
    def __init__(self):
        self.coefficients = np.array([[]])
        self.constraints = np.array([[]])
        self.basis_coefficients = np.array([[]])
        self.basis_solution = np.array([[]])
        self.basis_value = 0
        self.value = 0
        self.solution = np.array([[]])
        self.reduced_cost = np.array([[]])
        self.least_positive_ratio = np.array([[]])

    def initialize_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        basis_size = self.coefficients.shape[0]
        self.coefficients = np.append(self.coefficients, np.identity(basis_size), axis=1)

    def initialize_basis(self):
        """Sets up the initial basis."""
        self.basis_solution = self.constraints
        self.basis_coefficients = np.zeros(self.constraints.shape)