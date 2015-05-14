"""The display for the simplex running."""
from simplex import Simplex

from numbers import Number
from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Use latex.
mpl.rc('text', usetex=True)
custom_preamble = {
    "text.latex.preamble": ["\\usepackage{tabularx, colortbl, xcolor, color}"]
}
mpl.rcParams.update(custom_preamble)

blue = r" \cellcolor{blue!25} "
star = r" $\star$ "

def number_to_latex_display_string(number):
    if abs(number) == float('inf'):
        return r"$\infty$"
    fraction = Fraction(number).limit_denominator(10000)
    if fraction.denominator == 1:
        return r"$" + str(fraction.numerator) + r"$"
    elif fraction.denominator == 0:
        return r"$\infty$"
    else:
        return r"$\frac{" + str(fraction.numerator) + r"}{" + str(fraction.denominator) + r"}$"

def dn(number):
    return number_to_latex_display_string(number)

class Display:
    def __init__(self, simplex_init=Simplex()):
        self.simplex = simplex_init
        self.simplex.initialize_tableau()
        self.number_of_variables = self.simplex.coefficients.shape[1]
        self.number_of_columns = self.number_of_variables + 4
        self.number_of_basis_variables = self.simplex.basis_size
        self.figure = plt.figure(figsize=(18, 9), dpi=80)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.text = plt.text(0, 0, '', fontsize=40)
        self.initial_draw_done = False
        self.color_dict = {}
        self.clear_colors()
        self.fig_count = 0

    def clear_colors(self):
        co = []
        for _ in range(self.simplex.basis_size):
            co.append([""] * self.number_of_variables)
        self.color_dict = {
            "objective": [""] * self.number_of_variables,
            "cb": [""] * self.simplex.basis_size,
            "xb": [""] * self.simplex.basis_size,
            "bv": [""] * self.simplex.basis_size,
            "variables": [""] * self.number_of_variables,
            "coefficients": co,
            "reduced": [""] * self.number_of_variables,
            "ratio": [""] * self.simplex.basis_size,
            "value": ""
        }

    def run_simplex(self):
        """Run simplex with display."""
        # Display the starting tableau.
        self.display_tableau()
        self.simplex.calculate_basis_value()
        while True:
            # Calculate reduced costs.
            self.simplex.calculate_reduced_costs()
            self.color_dict['reduced'] = [star for _ in self.color_dict['reduced']]
            self.display_tableau()
            # End if the solution is optimal or unbounded.
            if self.simplex.check_if_optimal():
                self.simplex.obtain_solution()
                self.simplex.value = np.inner(self.simplex.objective[0:self.simplex.number_of_variables],
                                              self.simplex.solution.flatten())
                self.display_optimal()
                return
            if self.simplex.check_if_unbounded():
                self.simplex.value = float('inf')
                self.display_unbounded()
                return
            # Determine the pivot.
            self.simplex.obtain_pivot_column_index()
            self.simplex.obtain_pivot_row_index()
            self.color_dict['reduced'][self.simplex.pivot_column_index] = star
            self.color_dict['ratio'] = [star for _ in self.color_dict['ratio']]
            self.display_tableau()
            self.color_dict['reduced'][self.simplex.pivot_column_index] = star
            self.color_dict['ratio'][self.simplex.pivot_row_index] = star
            self.color_dict['coefficients'][self.simplex.pivot_row_index][self.simplex.pivot_column_index] = star
            self.display_tableau()
            # Perform pivot.
            self.simplex.make_pivot_element_one()
            self.simplex.make_pivot_independent()
            self.color_dict['coefficients'][self.simplex.pivot_row_index][self.simplex.pivot_column_index] = star
            self.display_tableau()
            self.simplex.swap_basis_variable()
            self.color_dict['bv'][self.simplex.pivot_row_index] = star
            self.color_dict['cb'][self.simplex.pivot_row_index] = star
            self.color_dict['variables'][self.simplex.pivot_column_index] = star
            self.color_dict['objective'][self.simplex.pivot_column_index] = star
            self.simplex.calculate_basis_value()
            self.color_dict['value'] = star
            self.display_tableau()

    def display_tableau(self):
        """Show only the tableau."""
        latex = self.attain_tableau_latex()
        self.display_latex(latex)

    def attain_tableau_latex(self):
        number_of_variables = self.number_of_variables
        number_of_columns = self.number_of_columns
        number_of_basis_variables = self.number_of_basis_variables
        c = self.color_dict

        # Get the column settings.
        column_settings = "| X | X X |"
        for _ in range(number_of_variables):
            column_settings += " X"
        column_settings += " | X |"

        # Setup the objective row.
        objective_row = r"\cline{4-" + str(number_of_columns - 1) + r"} \multicolumn{2}{c}{} & $c_j$"
        for index in range(number_of_variables):
            objective_row += r" & " + c['objective'][index] + dn(self.simplex.objective[index]) + r""
        objective_row += r" & \multicolumn{1}{r}{} \\ \cline{2-" + str(number_of_columns) + r"}"

        # Setup variable name row.
        variable_name_row = r"\multicolumn{1}{c|}{} & $c_b$ & $x_b$ "
        for index in range(number_of_variables):
            variable_name_row += r" & " + c['variables'][index]
            if index >= number_of_basis_variables:
                variable_name_row += r" $s_" + str(index - number_of_basis_variables + 1) + r"$"
            else:
                variable_name_row += r" $x_" + str(index + 1) + r"$"
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
            row += r" " + c['bv'][basis_index] + r" "
            if variable.is_slack:
                row += r"$s_"
            else:
                row += r"$x_"
            row += str(variable.number + 1) + r"$ & "
            row += r" " + c['cb'][basis_index] + r" " + dn(objective) + r" & " + dn(solution)
            for i, coefficient in enumerate(coefficients):
                row += r" & " + c['coefficients'][basis_index][i] + dn(coefficient)
            row += r" & " + (c['ratio'][basis_index] + dn(ratio) if isinstance(ratio, Number) else r"") + r" \\ "
            main_rows.append(row)
        main_rows[-1] += "\hline "

        # Setup reduced cost row.
        reduced_cost_row = r"\multicolumn{1}{c}{"
        reduced_cost_row += ((c['value'] + r"$c_b x_b = $" + dn(self.simplex.basis_value))
                             if isinstance(self.simplex.basis_value, Number) else r"")
        reduced_cost_row += r"} & & "
        reduced_cost_row += r"$\bar{c_j}$"
        for index in range(number_of_variables):
            try:
                reduced_cost = self.simplex.reduced_costs[index]
            except IndexError:
                reduced_cost = None
            reduced_cost_row += r" & " + (c['reduced'][index] + dn(reduced_cost) if isinstance(reduced_cost, Number) else r"") + r""
        reduced_cost_row += r" & \multicolumn{1}{c}{} \\ \cline{4-" + str(number_of_columns - 1) + r"}"

        latex = r"""{\renewcommand{\arraystretch}{1.2}"""
        latex += r"""\begin{tabularx}{1100pt}{""" + column_settings + r"""}""" + objective_row
        latex += variable_name_row
        for row in main_rows:
            latex += row
        latex += reduced_cost_row
        latex += r"""\end{tabularx}"""
        latex += r"""}"""
        latex.replace('\n', '')

        return latex

    def save_fig(self):
        number = str(self.fig_count).zfill(3)
        plt.savefig('out' + number + ".svg")
        self.fig_count += 1

    def display_latex(self, latex):
        self.text.set_text(latex)
        if self.initial_draw_done:
            while not plt.waitforbuttonpress():
                pass
            plt.draw()
        else:
            self.initial_draw_done = True
            plt.show(block=False)
        self.clear_colors()

    def display_optimal(self):
        tableau_latex = self.attain_tableau_latex()
        optimal_latex = r"""\noindent Optimal value: """ + dn(self.simplex.value) + r" \\ "
        optimal_latex += r"""Solution: """
        for i, s in enumerate(self.simplex.solution):
            if i != 0:
                optimal_latex += r", "
            optimal_latex += r"$x_" + str(i + 1) + r" = \,$" + dn(s[0])
        optimal_latex += r" \\"
        self.text.set_text(optimal_latex + tableau_latex)
        plt.waitforbuttonpress()
        plt.draw()
        plt.waitforbuttonpress()

    def display_unbounded(self):
        tableau_latex = self.attain_tableau_latex()
        unbounded_latex = r"""\noindent Unbounded!"""
        unbounded_latex += r" \\"
        self.text.set_text(unbounded_latex + tableau_latex)
        plt.waitforbuttonpress()
        plt.draw()
        plt.waitforbuttonpress()

if __name__ == "__main__":
    coefficients = np.array([[1,  1],
                             [1, -1]], dtype='float')
    constraints = np.array([[4],
                            [2]], dtype='float')
    objective = np.array([3, 2])
    simplex = Simplex(coefficients=coefficients, constraints=constraints, objective=objective)
    display = Display(simplex_init=simplex)
    display.run_simplex()
