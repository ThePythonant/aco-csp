# -*- coding: utf-8 -*-
"""Closest String Problem solution."""
import numpy as np
import logging

logging_level = logging.INFO
logger = logging.getLogger('CSP Solver')
logger.setLevel(logging_level)
ch = logging.StreamHandler()
ch.setLevel(logging_level)
logger.addHandler(ch)


def hamming_distance(str1, str2):
    """Computes the hamming distance between two strings."""
    if len(str1) != len(str2):
        logger.error("Strings should have the same lenght"
                     " for computing the hamming distance")
        raise Exception
    hd = 0
    for c1, c2 in zip(str1, str2):
        hd += hamming_function(c1, c2)
    return hd


def hamming_function(char1, char2):
    """Helper function to compute the Hamming distance between 2 characters."""
    if char1 == char2:
        return 0
    return 1


v_hamming_distance = np.vectorize(hamming_distance)


def pick_random(arr, ran):
    """Wrapper function for random choice to allow vectorization."""
    return np.random.choice(ran, p=arr)


v_pick_random = np.vectorize(pick_random)


class Problem(object):
    """Definition of a general purpose ACO problem."""

    def __init__(self, location):
        self._init_internal_structures()
        self.load_input_data(location)

    def load_input_data(self, location):
        """Load input data from external instance file."""
        raise NotImplementedError

    def _init_internal_structures(self):
        """Initialise specific structures to be used by the problem."""
        raise NotImplementedError


class Solver(object):
    """Definition of a general purpose ACO solver."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._init_problem()
        self.alpha = self.cfg['--alpha']
        self.beta = self.cfg['--beta']
        if self.cfg['--rho']:
            self.rho = float(self.cfg['--rho'])
        self.num_ants = int(self.cfg['--numants'])
        self.max_iter = int(self.cfg['--maxiter'])
        self.init_pheromone()
        self.init_ants()

    def solve(self):
        """Method in charge of solving the problem."""
        raise NotImplementedError

    def _init_problem(self):
        """Initialise the problem to be solved by the solver."""
        raise NotImplementedError

    def init_pheromone(self):
        """Initialise the pheromone information."""
        raise NotImplementedError

    def init_ants(self):
        """Initialise the ants that will form the colony."""
        raise NotImplementedError

    def terminate(self):
        """Give the termination condition."""
        raise NotImplementedError

    def normalize_pheromone(self):
        self.pheromone[self.pheromone < 0] = 0
        self.pheromone = self.pheromone / self.pheromone.sum(axis=0)
        logger.debug("Normalized pheromone: %s" % self.pheromone)


class CSPProblem(Problem):
    """Closest String Problem Representation."""

    def __init__(self, instance_location):
        super(CSPProblem, self).__init__(instance_location)
        self.strings = np.array(self.strings)
        self.inv_alphabet = {ch: i for i, ch in enumerate(self.alphabet)}

    def _init_internal_structures(self):
        self.strings = []
        self.alphabet = []

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


class CSPSolver(Solver):
    """Closest String Problem ACO Solver class.

    Contains the colony and main structure for solving the problem."""

    def __init__(self, cfg):
        super(CSPSolver, self).__init__(cfg)

    def _init_problem(self):
        """Assign CSP Problem to the solver."""
        self.problem = CSPProblem(self.cfg['--instance'])

    def solve(self):
        """Solve method for the CSP."""
        self._num_evaluations = 0
        while not self.terminate():
            for ant in self.ants:
                ant.find_solution(self.pheromone, self.problem.alphabet)
                ant.evaluate_solution(self.problem.strings)
                if self.best_ant.score > ant.score:
                    self.best_ant = ant
            self.evaporate_pheromone()
            self.deposit_pheromone()
            self.normalize_pheromone()
            self._num_evaluations += 1
            logger.info('Iteration: %d score: %d' % (self._num_evaluations,
                                                     self.best_ant.score))

    def terminate(self):
        """Set termination condition."""
        return self._num_evaluations >= self.max_iter

    def init_ants(self):
        """Initialise the colony members."""
        self.ants = []
        for i in range(self.num_ants):
            ant = Ant()
            ant.find_solution(self.pheromone, self.problem.alphabet)
            ant.evaluate_solution(self.problem.strings)
            if i == 0:
                self.best_ant = ant
            elif self.best_ant.score > ant.score:
                self.best_ant = ant
            self.ants.append(ant)
        self.evaporate_pheromone()
        self.deposit_pheromone()
        self.normalize_pheromone()

    def init_pheromone(self):
        """Pheromone initialised with a constant value 1/|Alphabet|."""
        self.pheromone = np.empty(
            (self.problem.alphabet_length, self.problem.str_length),
            dtype=float)
        self.pheromone.fill(1 / self.problem.alphabet_length)

    def evaporate_pheromone(self):
        """Evaporation of pheromone."""
        self.pheromone -= self.rho

    def deposit_pheromone(self):
        """Deposit pheromone into the pheromone data structure."""
        for i, ch in enumerate(self.best_ant.solution):
            self.pheromone[self.problem.inv_alphabet[ch]][i] += (
                1.0 -
                self.best_ant.max_hamming_distance / self.problem.str_length)


class Ant(object):
    """Ant definition for the CSP."""

    def __init__(self):
        pass

    def _rnd_choice(self, arr, ran):
        return np.random.choice(ran, p=arr)

    def find_solution(self, pheromone, alphabet):
        logger.debug("Pheromone to be used in the solution: %s" % pheromone)
        str_pos = np.apply_along_axis(self._rnd_choice, 0, pheromone,
                                      len(alphabet))
        self.solution = ''.join([alphabet[p] for p in str_pos])

    def evaluate_solution(self, strings):
        """Evaluates the current solution distance to the strings parameter."""
        hamming_distances = v_hamming_distance(strings, self.solution)
        self.max_hamming_distance = np.amax(hamming_distances)
        self.score = hamming_distances.sum()
