import numpy as np


class FitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._score_matrix = None

    def __call__(self, ind):
        ind_semantics = self.get_semantics(ind)
        raw_fitness_vector = self.get_raw_fitness(ind_semantics)
        shared_fitness_vector = self.get_shared_fitness(ind_semantics, raw_fitness_vector)
        self.register_semantics(ind_indsemantics)
        return np.sum(raw_fitness_vector / shared_fitness_vector)

    def get_semantics(self, ind):
        v_ind = np.vectorize(ind)
        return v_ind(self._cases[0])

    def get_raw_fitness(self, ind_semantics):
        raise NotImplementedError()

    def get_shared_fitness(self, ind_semantics, error_vector):
        if self._score_matrix:
            comparrison_matrix = ind_semantics == self._score_matrix
            rv = np.sum(comparrision_matrix.astype(np.int64) * error_vector, axis=0)
            rv[rv == 0] = 1
            return rv
        else:
            return np.ones(ind_sematics.size)

    def register_semantics(self, ind_semantics):
        if self._score_matrix:
            self._score_matrix = np.vstack((self._score_matrix, ind_semantics))
        else:
            self._score_matrix = np.array([ind_semantics])
