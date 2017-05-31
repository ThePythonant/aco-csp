# -*- coding: utf-8 -*-
"""Closest String Problem solution."""


class Problem(object):
    """Definition of a general purpose ACO problem."""

    def __init__(self, location):
        self.load_input_data(location)

    def load_input_data(self, location):
        raise NotImplementedError


class Solver(object):
    """Definition of a general purpose ACO solver."""

    def __init__(self, cfg):
        self.alpha = cfg['--alpha']
        self.beta = cfg['--beta']
        self.num_ants = cfg['--numants']


class CSPProblem(Problem):
    """Closest String Problem Representation."""

    def __init__(self, instance_location):
        self.strings = []
        self.alphabet = []
        super(CSPProblem, self).__init__(instance_location)

    def load_input_data(self, instance_location):
        """Load input data from provided instances format."""
        with open(instance_location) as file:
            for i, line in enumerate(file):
                if i == 0:
                    self.alphabet_length = int(line.strip())
                elif i == 1:
                    self.num_str = int(line.strip())
                elif i == 2:
                    self.str_length = int(line.strip())
                elif i <= self.alphabet_length + 2:
                    self.alphabet.append(line.strip())
                elif line.strip():
                    self.strings.append(line.strip())

