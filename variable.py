"""Simplex variables identifiers."""


class Variable:
    """A class for identifying a simplex variable."""
    def __init__(self, index=0, is_slack=True):
        self.number = index
        self.is_slack = is_slack

    def __eq__(self, other):
        return self.number == other.number and self.is_slack == other.is_slack