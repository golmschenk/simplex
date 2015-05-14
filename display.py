"""The display for the simplex running."""
from simplex import Simplex

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Use latex.
mpl.rc('text', usetex=True)


class Display:
    def __init__(self):
        self.simplex = Simplex()

    def run_simplex(self):
        """Run simplex with display."""
        # Set up the tableau.
        self.simplex.initialize_tableau()
        # Display the starting tableau.
        self.simplex.display_tableau()
        while True:
            # Calculate the value and reduced costs.
            self.simplex.calculate_basis_value()
            self.simplex.calculate_reduced_costs()
            # End if the solution is optimal or unbounded.
            if self.simplex.check_if_optimal():
                self.simplex.value = self.simplex.basis_value
                self.simplex.obtain_solution()
                return
            if self.simplex.check_if_unbounded():
                self.simplex.value = float('inf')
                return
            # Determine the pivot.
            self.simplex.obtain_pivot_column_index()
            self.simplex.obtain_pivot_row_index()
            # Perform pivot.
            self.simplex.make_pivot_element_one()
            self.simplex.make_pivot_independent()
            self.simplex.swap_basis_variable()

    def display_tableau(self):
        """Show only the tableau."""
        latex = self.attain_tableau_latex()
        self.display_latex(latex)

    def attain_tableau_latex(self):
        number_of_variables = self.simplex.coefficients.shape[1]
        number_of_columns = number_of_variables + 4
        number_of_basis_variables = self.simplex.basis_size

        # Get the column settings.
        column_settings = "| c | c c |"
        for _ in range(number_of_variables):
            column_settings += " c"
        column_settings += " | c |"

        # Setup the objective row.
        objective_row = r"\cline{4-" + str(number_of_columns - 1) + r"} \multicolumn{2}{c}{} & $c_j$"
        for index in range(number_of_variables):
            objective_row += r" & " + str(self.simplex.objective[index]) + r""
        objective_row += r" & \multicolumn{1}{r}{} \\ \cline{2-" + str(number_of_columns) + r"}"

        # Setup variable name row.
        variable_name_row = r"\multicolumn{1}{c|}{} & $c_b$ & $x_b$"
        for index in range(number_of_variables):
            if index >= number_of_basis_variables:
                variable_name_row += r" & $s_" + str(index - number_of_basis_variables + 1) + r"$"
            else:
                variable_name_row += r" & $x_" + str(index + 1) + r"$"
        variable_name_row += r" & $\frac{x_b}{x_i}$ \\ \hline "

        # Setup main rows.
        main_rows = []
        for basis_index in range(self.simplex.basis_size):
            row = r""
            variable = self.simplex.basis_variables[basis_index]
            objective = self.simplex.basis_objective[basis_index][0]
            solution = self.simplex.basis_solution[basis_index][0]
            coefficients = self.simplex.coefficients[basis_index].flatten()
            try:
                ratio = self.simplex.least_positive_ratio[basis_index]
            except IndexError:
                ratio = None
            if variable.is_slack:
                row += r"$s_"
            else:
                row += r"$x_"
            row += str(variable.number + 1) + r"$ & " + str(objective) + r" & " + str(solution)
            for coefficient in coefficients:
                row += r" & " + str(coefficient)
            row += r" & " + (str(ratio) if ratio else r"") + r" \\ "
            main_rows.append(row)
        main_rows[-1] += "\hline "

        # Setup reduced cost row.
        reduced_cost_row = r"\multicolumn{1}{c}{"
        reduced_cost_row += ((r"$c_b x_b = " + str(simplex.basis_value) + r"$") if simplex.basis_value else r"")
        reduced_cost_row += r"} & & "
        reduced_cost_row += r"$\bar{c_j}$"
        for index in range(number_of_variables):
            try:
                reduced_cost = self.simplex.reduced_costs[index]
            except IndexError:
                reduced_cost = None
            reduced_cost_row += r" & " + (str(reduced_cost) if reduced_cost else r"") + r""
        reduced_cost_row += r" & \multicolumn{1}{| c}{} \\ \cline{4-" + str(number_of_columns - 1) + r"}"


        latex = r"""\begin{tabular}{""" + column_settings + r"""}""" + objective_row
        latex += variable_name_row
        for row in main_rows:
            latex += row
        latex += reduced_cost_row
        latex += r"""\end{tabular}"""
        latex.replace('\n', '')
        return latex

    def display_latex(self, latex):
        plt.text(0, 0,'%s' % latex, fontsize=40)
        plt.show()

if __name__ == "__main__":
    coefficients = np.array([[1,  1],
                             [1, -1]], dtype='float')
    constraints = np.array([[4],
                            [2]], dtype='float')
    objective = np.array([3, 2])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display()
    display.simplex = simplex
    display.simplex.initialize_tableau()
    display.display_tableau()