import numpy as np
from fitness_sharing_function import FitnessSharingFunction


def test_init():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y = [(0,), (1,), (1,), (0,)]

    EXPECTED_CASES = [X, y]
    EXPECTED_SCORE_MATRIX = np.zeros((1, len(y)))

    fsf = FitnessSharingFunction(X, y)

    assert fsf._cases == EXPECTED_CASES
    assert fsf._score_matrix == EXPECTED_SCORE_MATRIX
