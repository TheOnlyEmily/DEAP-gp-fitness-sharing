import numpy as np


class FitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._semantic_matrix = None

    def __call__(self, ind):
        base_rewards = self.get_reward(self.get_semantics(ind))
        reward_adjust = self.get_shared_fitness(ind)
        adjusted_rewards = map(lambda v: v / reward_adjust, base_rewards)
        self.register_semantics(ind)
        return sum(adjusted_rewards)

    def get_reward(self, ind_semantics):
        raise NotImplementedError()

    def get_semantics(self, ind):
        v_ind = np.vectorize(ind)
        return v_ind(self._cases[0])

    def get_shared_fitness(self, ind_semantics):
        if self._semantic_matrix:
            weight_vector = np.sum(ind_semantics == self._semantic_matrix, axis=0)
            fit_adjust = np.sum(weight_vector * self.get_reward(ind))
            return fit_adjust if fit_adjust > 0 else 1
        else:
            return 1

    def register_semantics(self, ind):
        if self._semantic_matrix:
            ind_semantics = self.get_semantics(ind)
            semantic_stack = (self._semantic_matrix, ind_semantics)
            self._semantic_matrix = np.vstack(semantic_stack)
        else:
            self._semantic_matrix = np.array([self.get_semantics(ind)])
