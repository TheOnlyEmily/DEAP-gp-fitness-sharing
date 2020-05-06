from itertools import starmap

import numpy as np


class SemanticFitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._semantic_matrix = None

    def __call__(self, ind):
        base_reward = self.get_fitness(self.get_semantics(ind))
        reward_adjust = self.get_shared_fitness(ind)
        self.register_semantics(ind)
        return base_reward * reward_adjust

    @property
    def target_semantics(self):
        return self._cases[1].copy()

    def get_fitness(self, ind_semantics):
        raise NotImplementedError()

    def get_semantics(self, ind):
        return np.array(starmap(ind, self._cases[0]))

    def get_shared_fitness(self, ind):
        if self._semantic_matrix is not None:
            ind_semantics = self.get_semantics(ind)
            weight_vector = np.sum(ind_semantics == self._semantic_matrix, axis=0)
            fit_adjust = np.sum(weight_vector * self.get_fitness(ind_semantics))
            return fit_adjust if fit_adjust > 0 else 1
        else:
            return 1

    def register_semantics(self, ind):
        if self._semantic_matrix is not None:
            ind_semantics = self.get_semantics(ind)
            semantic_stack = (self._semantic_matrix, ind_semantics)
            self._semantic_matrix = np.vstack(semantic_stack)
        else:
            self._semantic_matrix = np.array([self.get_semantics(ind)])
