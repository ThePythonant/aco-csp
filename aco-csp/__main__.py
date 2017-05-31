# -*- coding: utf-8 -*-
"""
Closest String Problem ACO Solver.

Usage:
  aco_csp [options]

Options:
    -i --instance=<path>  Path to instance file
"""
from docopt import docopt
from pprint import pprint
from csp_solver import CSPProblem

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')
    prob = CSPProblem(arguments['instance'])
