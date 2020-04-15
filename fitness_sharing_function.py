import numpy as np


class FitnessSharingFunction:

    def __init__(self, X, y):
        self._cases = [X, y]
        self._semantic_matrix = None

    def __call__(self, ind):
        base_rewards = map(self.get_reward, self.get_semantics(ind))
        self._adjusted_reward = sum(self.get_shared_fitness(ind))
        modified_rewards = map(self.reward_adjust, base_rewards)
        self.register_semantics(ind)
        return sum(modified_rewards)

    def get_reward(self, ind):
        raise NotImplementedError()

    def get_semantics(self, ind):
        v_ind = np.vectorize(ind)
        return v_ind(self._cases[0])

    def get_shared_fitness(self, ind):
        if self._semantic_matrix:
            weight_vector = np.sum(ind == self._semantic_matrix, axis=0)
            return weight_vector * self.get_reward(ind)
        else:
            return np.ones(ind.size) / ind.size

    def register_semantics(self, ind):
        if self._semantic_matrix:
            ind_semantics = self.get_semantics(ind)
            semantic_stack = (self._semantic_matrix, ind_semantics)
            self._semantic_matrix = np.vstack(semantic_stack)
        else:
            self._semantic_matrix = np.array([self.get_semantics(ind)])

    def reward_adjust(self, reward):
        if self._adjusted_reward > 0:
            return reward / self._adjusted_reward
        else:
            return reward
