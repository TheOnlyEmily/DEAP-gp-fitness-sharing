from itertools import starmap

import numpy as np


class SemanticFitnessSharingFunction:

    def __init__(self):
        self._delta_error_matrix = None

    def __call__(self, delta_error):
        self.register_error_vector(delta_error)
        return self.get_shared_fitness(delta_error)

    def get_shared_fitness(self, delta_error):
        if self._delta_error_matrix is not None:
            weight_vector = np.sum(delta_error == self._delta_error_matrix, axis=0)
            fit_adjust = np.sqrt(np.sum((delta_error * weight_vector)**2))
            return fit_adjust if fit_adjust > 0 else 1
        else:
            return 1

    def register_error_vector(self, delta_error):
        if self._delta_error_matrix is not None:
            delta_error_stack = (self._delta_error_matrix, delta_error)
            self._delta_error_matrix = np.vstack(delta_error_stack)
        else:
            self._delta_error_matrix = np.array([np.array(detla_error)])
