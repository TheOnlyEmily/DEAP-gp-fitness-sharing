import numpy as np
import pytest
from fitness_sharing_function import FitnessSharingFunction


def test_init():
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 0])

    CASE_LIST = [X, y]

    fsf = FitnessSharingFunction(X, y)

    assert fsf._cases == CASE_LIST
    assert fsf._semantic_matrix is None

def test_get_semantics():
    from operator import xor

    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 0])

    fsf = FitnessSharingFunction(X, y)

    assert np.all(fsf.get_semantics(xor) == y)


class TestGetReward:

    def test_method_not_implemented(self):
        X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        y = np.array([0, 1, 1, 0])

        function_semantics = np.array([0, 0, 0, 1])

        fsf = FitnessSharingFunction(X, y)

        with pytest.raises(NotImplementedError):
            fsf.get_reward(function_semantics)


    def test_method_implemented(self):
        X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        y = np.array([0, 1, 1, 0])

        perfect_solution = np.array([0, 1, 1, 0])
        okay_solution = np.array([0, 1, 0, 0])

        class SemDistanceFSF(FitnessSharingFunction):

            def get_reward(self, ind_semantics):
                return np.sum((self._cases[1] - ind_semantics) ** 2)

        fsf = SemDistanceFSF(X, y)

        assert np.all(fsf.get_reward(perfect_solution) == 0)
        assert np.all(fsf.get_reward(okay_solution) > 0)


def test_register_semantics():
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 0])

    IND1 = lambda a, b: a & b
    IND2 = lambda a, b: a ^ b

    fsf = FitnessSharingFunction(X, y)

    IND_SEMANTICS1 = fsf.get_semantics(IND1)
    IND_SEMANTICS2 = fsf.get_semantics(IND2)

    assert fsf._semantic_matrix is None

    fsf.register_semantics(IND1)

    assert np.all(fsf._semantic_matrix == np.array([IND_SEMANTICS1]))

    fsf.register_semantics(IND2)

    assert np.all(fsf._semantic_matrix == np.array([IND_SEMANTICS1, IND_SEMANTICS2]))


class TestGetSharedFitness:

    def test_with_no_semantic_matrix(self):
        X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        y = np.array([0, 1, 1, 0])


        class SemDistanceFSF(FitnessSharingFunction):

            def get_reward(self, ind_semantics):
                return np.sum((self._cases[1] - ind_semantics) ** 2)


        fsf = SemDistanceFSF(X, y)

        NO_ADJUSTMENT = 1
        IND = lambda a, b: a & b

        assert fsf._semantic_matrix is None
        assert fsf.get_shared_fitness(IND) == NO_ADJUSTMENT

    def test_with_semantic_matrix(self):
        X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        y = np.array([0, 1, 1, 0])


        class SemDistanceFSF(FitnessSharingFunction):

            def get_reward(self, ind_semantics):
                return np.sum((self._cases[1] - ind_semantics) ** 2)


        fsf = SemDistanceFSF(X, y)

        IND = lambda a, b: a & b
        IND_SEMANTICS = np.array([0, 0, 0, 1])

        fsf.register_semantics(IND)

        assert fsf.get_shared_fitness(IND_SEMANTICS) < 1


def test_call():
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    y = np.array([0, 1, 1, 0])


    class SemDistanceFSF(FitnessSharingFunction):

        def get_reward(self, ind_semantics):
            return np.sum((self._cases[1] - ind_semantics) ** 2)


    fsf = SemDistanceFSF(X, y)

    IND1 = lambda a, b: a & b
    IND2 = lambda a, b: a & b
    IND3 = lambda a, b: a ^ b

    fitness1 = fsf(IND1)
    fitness2 = fsf(IND2)
    fitness3 = fsf(IND3)

    assert fitness1 == fsf.get_reward(fsf.get_semantics(IND1))
    assert fitness1 < fitness2
    assert fitness3 < fitness1
    assert fitness3 == 0
