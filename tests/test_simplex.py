"""Tests for the simplex module."""
import numpy as np
from simplex import Simplex


class TestSimplex:
    """Tests for the simplex class."""
    def test_can_add_slack_variables_to_matrix(self):
        """Checks that the slack variables can be added correctly."""
        A = np.array([[1,  1],
                      [1, -1]])
        expected_A_with_slack = np.array([[1,  1, 1, 0],
                                          [1, -1, 0, 1]])

        A_with_slack = Simplex.add_slack(A)

        assert A_with_slack == expected_A_with_slack

    def test_can_make_tableau_from_constraints(self):
        # A = np.array([[]])
        assert False # Finish me.