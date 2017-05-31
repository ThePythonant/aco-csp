# -*- coding: utf-8 -*-
"""ACO CSP module general components."""


class Problem(object):
    """Definition of a general purpose ACO problem."""

    def __init__(self, location):
        self.load_input_data(location)

    def load_input_data(self, location):
        raise NotImplementedError


class Solver(object):
    """Definition of a general purpose ACO solver."""
