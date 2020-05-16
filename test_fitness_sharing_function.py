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

    assert np.all(fsf._semantic_matrix == np.array([DELTA_ERROR1]))

    fsf.register_semantics(DELTA_ERROR2)

    assert np.all(fsf._semantic_matrix == np.array([DELTA_ERROR1, DELTA_ERROR2]))


class TestGetSharedFitness:

    def test_with_no_semantic_matrix(self):
        fsf = SemDistanceFSF(X, y)

        NO_ADJUSTMENT = 1
        IND = lambda a, b: a & b

        assert fsf._semantic_matrix is None
        assert fsf.get_shared_fitness(IND) == NO_ADJUSTMENT

    def test_with_semantic_matrix(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])


        class SemDistanceFSF(SemanticFitnessSharingFunction):

            def get_fitness(self, ind_semantics):
                return np.mean((self.target_semantics - ind_semantics) ** 2)


        fsf = SemDistanceFSF(X, y)

        IND = lambda a, b: a & b
        IND_SEMANTICS = np.array([0, 0, 0, 1])

        fsf.register_semantics(IND)

        assert fsf.get_shared_fitness(IND) > 1


def test_call():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])


    class SemDistanceFSF(SemanticFitnessSharingFunction):

        def get_fitness(self, ind_semantics):
            return np.mean((self.target_semantics - ind_semantics) ** 2)


    fsf = SemDistanceFSF(X, y)

    IND1 = lambda a, b: a & b
    IND2 = lambda a, b: a & b
    IND3 = lambda a, b: a ^ b

    fitness1 = fsf(IND1)
    fitness2 = fsf(IND2)
    fitness3 = fsf(IND3)

    assert fitness1 == fsf.get_fitness(fsf.get_semantics(IND1))
    assert fitness1 < fitness2
    assert fitness3 < fitness1
    assert fitness3 == 0
