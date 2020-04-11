import numpy as np


class FitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._score_matrix = np.zeros((len(y), 1))

    def __call__(self, ind):
        ind_semantics = self.get_semantics(ind)
        raw_fitness_vector = self.get_raw_fitness(ind_semantics)
        shared_fitness_vector = self.get_shared_fitness(ind_semantics)
        self.register_semantics(ind_indsemantics)
        return np.sum(raw_fitness_vector / shared_fitness_vector)

    def get_semantics(self, ind):
        return np.array([ind(*X) for X, _ in self._cases])

    def get_raw_fitness(self, ind_semantics):
        raise NotImplementedError()

    def get_shared_fitness(self, ind_semantics):
        semantic_matrix = np.tile(ind_semantics.reshape((ind_semantics.size, 1)), (ind_semantics.size, self._score_matrix.shape[1]))
        comparrison_matrix = semantic_matrix == self._score_matrix
        return np.sum(comparrision_matrix, axis=1).reshape((1, comparrison_matrix.shape[0]))
