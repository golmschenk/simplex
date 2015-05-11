"""
The class for solving simplex.
Given a standard LPP, finds the optimal solution through simplex method.
"""

import numpy as np

class Simplex:
    """Class to preform simplex."""
    def __init__(self):
        self.A = np.array([[]])
        self.b = np.array([[]])

    def add_slack(self):
        """Adds the slack identity matrix to the A matrix."""
        identity_size = self.A.shape[0]
        self.A = np.append(self.A, np.identity(identity_size), axis=1)
