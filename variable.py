"""Simplex variables identifiers."""


class Variable:
    """A class for identifying a simplex variable."""
    def __init__(self):
        self.number = 0
        self.type = "slack"