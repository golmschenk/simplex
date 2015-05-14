"""Run the main program."""
import numpy as np
from simplex import Simplex
from display import Display
from examples import example1, example2, example3, example4

def run_from_txt():
    with open("lp.txt") as file:
        content = file.read()
    kw = {}
    exec(content, globals(), kw)
    A = np.array(kw['A'])
    b = np.array(kw['b'])
    c = np.array(kw['c'])
    d = Display(Simplex(coefficients=A, constraints=b, objective=c))
    d.run_simplex()

run_from_txt()
#example1()
#example2()
#example3()
#example4()