# -*- coding: utf-8 -*-
"""
Closest String Problem ACO Solver.

Usage:
  aco_csp [options]

Options:
  -i --instance=<path>  Path to instance file
  -a --alpha=<val>      Alpha value to be used
  -b --beta=<val>       Beta value to be used
  -r --rho=<val>        Rho value to be used [default: 0.003]
  -n --numants=<val>    Number of ants [default: 10]
  -s --seed=<val>       Seed to use in the random generator [default: 1234]
  -m --maxiter=<val>    Maximum number of iterations to run [default: 1000]
"""
from docopt import docopt
from csp_solver import CSPSolver
import numpy as np

if __name__ == '__main__':
    args = docopt(__doc__, version='FIXME')
    np.random.seed(int(args['--seed']))
    solver = CSPSolver(args)
    solver.solve()
