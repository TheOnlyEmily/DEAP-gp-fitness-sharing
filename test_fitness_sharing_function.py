import numpy as np
import pytest
from fitness_sharing_function import SemanticFitnessSharingFunction


def test_init():

    fsf = SemanticFitnessSharingFunction()

    assert fsf._delta_error_matrix is None

def test_register_error_vector():
    fsf = SemanticFitnessSharingFunction()

    DELTA_ERROR1 = np.array([0, 0, 0, 1]) - np.array([0, 1, 1, 0])
    DELTA_ERROR2 = np.array([0, 1, 1, 1]) - np.array([0, 1, 1, 0])

    assert fsf._delta_error_matrix is None

    fsf.register_error_vector(DELTA_ERROR1)

    assert np.all(fsf._delta_error_matrix == np.array([DELTA_ERROR1]))

    fsf.register_error_vector(DELTA_ERROR2)

    assert np.all(fsf._delta_error_matrix == np.array([DELTA_ERROR1, DELTA_ERROR2]))


class TestGetSharedFitness:

    def test_with_no_delta_error_matrix(self):
        fsf = SemanticFitnessSharingFunction()

        NO_ADJUSTMENT = 1
        DELTA_ERROR = np.array([0, 0, 0, 1]) - np.array([0, 1, 1, 0])

        assert fsf._delta_error_matrix is None
        assert fsf.get_shared_fitness(DELTA_ERROR) == NO_ADJUSTMENT

    def test_with_delta_error_matrix(self):
        fsf = SemanticFitnessSharingFunction()

        DELTA_ERROR = np.array([0, 0, 0, 1]) - np.array([0, 1, 1, 0])

        fsf.register_error_vector(DELTA_ERROR)

        assert fsf.get_shared_fitness(DELTA_ERROR) > 1


def test_call():
    fsf = SemanticFitnessSharingFunction()

    DELTA_ERROR1 = np.array([0, 1, 1, 0]) - np.array([0, 1, 1, 0])
    DELTA_ERROR2 = np.array([0, 0, 0, 1]) - np.array([0, 1, 1, 0])
    DELTA_ERROR3 = np.array([0, 0, 0, 1]) - np.array([0, 1, 1, 0])

    shared_f1 = fsf(DELTA_ERROR1)
    shared_f2 = fsf(DELTA_ERROR2)
    shared_f3 = fsf(DELTA_ERROR3)

    NO_ADJUSTMENT = 1

    assert shared_f1 == NO_ADJUSTMENT
    assert shared_f2 > shared_f1
    assert shared_f3 > shared_f2
