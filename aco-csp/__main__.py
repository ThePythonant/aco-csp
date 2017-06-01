# -*- coding: utf-8 -*-
"""
Closest String Problem ACO Solver.

Usage:
  aco_csp [options]

Options:
  -i --instance=<path>  Path to instance file
  -a --alpha=<val>      Alpha value to be used
  -b --beta=<val>       Beta value to be used
  -r --rho=<val>        Rho value to be used
  -n --numants=<val>    Number of ants [default: 10]
"""
from docopt import docopt
from csp_solver import CSPSolver

if __name__ == '__main__':
    args = docopt(__doc__, version='FIXME')
    solver = CSPSolver(args)
    solver.solve()
