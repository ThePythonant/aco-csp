# -*- coding: utf-8 -*-
"""Closest String Problem solution."""
from . import Problem


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


prob = CSPProblem('../instances/2-30-10000-1-9.csp')
print(prob.alphabet_length)
