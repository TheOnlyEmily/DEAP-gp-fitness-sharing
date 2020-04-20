import numpy as np
from fitness_sharing_function import FitnessSharingFunction


def test_init():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    CASE_LIST = [X, y]

    fsf = FitnessSharingFunction(X, y)

    assert fsf._cases == CASE_LIST
    assert fsf._semantic_matrix is None

def test_get_semantics():
    from operator import xor

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    fsf = FitnessSharingFunction(X, y)

    assert fsf.get_semantics(xor) == y
