import numpy as np


class FitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._score_matrix = np.zeros((1, len(y)))

    def __call__(self, ind):
        ind_semantics = self.get_semantics(ind)
        raw_fitness_vector = self.get_raw_fitness(ind_semantics)
        shared_fitness_vector = self.get_shared_fitness(ind_semantics, raw_fitness_vector)
        self.register_semantics(ind_indsemantics)
        shared_fitness_vector[shared_fitness_vector == 0] = 1
        return np.sum(raw_fitness_vector / shared_fitness_vector)

    def get_semantics(self, ind):
        v_ind = np.vectorize(ind)
        return v_ind(X)

    def get_raw_fitness(self, ind_semantics):
        raise NotImplementedError()

    def get_shared_fitness(self, ind_semantics, raw_fitness):
        semantic_matrix = np.tile(ind_semantics, self._score_matrix.shape)
        comparrison_matrix = semantic_matrix == self._score_matrix
        return np.sum(comparrision_matrix.astype(np.int64) * raw_fitness, axis=0)

    def register_semantics(self, ind_semantics):
        self._score_matrix = np.vstack(self._score_matrix, ind_semantics)
