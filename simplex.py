"""
The class for solving simplex.
Given a standard LPP, finds the optimal solution through simplex method.
"""

import numpy as np

class Simplex:
    """Class to preform simplex."""
    def __init__(self):
        self.coefficients = np.array([[]])
        self.b = np.array([[]])
        self.basis_size = 0

    def add_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        basis_size = self.coefficients.shape[0]
        self.basis_size = basis_size
        self.coefficients = np.append(self.coefficients, np.identity(basis_size), axis=1)
