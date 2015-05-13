"""Tests for the simplex variables."""

from variable import Variable


class TestVariable:
    """Tests for the variable class."""
    def test_variable_equality(self):
        """Test that variables can be checked for equality through the equality operator."""
        variable0 = Variable(index=2, is_slack=True)
        variable1 = Variable(index=2, is_slack=True)
        variable2 = Variable(index=1, is_slack=True)
        variable3 = Variable(index=2, is_slack=False)

        assert variable0 == variable1
        assert not variable0 == variable2
        assert not variable0 == variable3