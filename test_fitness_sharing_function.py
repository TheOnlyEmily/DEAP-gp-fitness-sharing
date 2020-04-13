import numpy as np
from fitness_sharing_function import FitnessSharingFunction


def test_init():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    EXPECTED_CASES = [X, y]
    EXPECTED_SCORE_MATRIX = np.zeros((1, len(y)))

    fsf = FitnessSharingFunction(X, y)

    assert fsf._cases == EXPECTED_CASES
    assert fsf._score_matrix == EXPECTED_SCORE_MATRIX

def test_get_semantics():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    EXPECTED_SEMANTICS = np.array([0, 0, 0, 1])

    fsf = FitnessSharingFunction(X, y)

    assert fsf.get_semantics(lambda a, b: a && b) == EXPECTED_SEMANTICS

def test_get_raw_fitness():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    fsf = FitnessSharingFunction(X, y)

    with pytest.raises(NotImplementedError):
        fsf.get_raw_fitness()


    class MyFitnessFunction(FitnessSharingFunction):

        def get_raw_fitness(self, ind_semantics):
            return np.mean(np.sqrt(self._cases[1]**2 - ind_semantics**2))


    y_error = np.array([0, 0, 0, 1])
    my_fsf = MyFitnessFunction(X, y)

    assert my_fsf.get_raw_fitness(y_error) == 3 / 4
    assert my_fsf.get_raw_fitness(y) == 0

def test_register_semantics():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    fsf = FitnessSharingFunction(X, y)

    NEW_SEMANTICS1 = np.array([0, 1, 0, 1])
    NEW_SEMANTICS2 = np.array([0, 0, 0, 1])
    SCORE_MATRIX1 = None
    SCORE_MATRIX2 = np.array([[0, 1, 0, 1]])
    SCORE_MATRIX3 = np.array([[0, 1, 0, 1], [0, 0, 0, 1]])

    assert fsf._score_matrix == SCORE_MATRIX1

    fsf.register_semantics(NEW_SEMANTICS1)

    assert fsf._score_matrix == SCORE_MATRIX2

    fsf.register_semantics(NEW_SEMANTICS2)

    assert fsf._score_matrix == SCORE_MATRIX3

def test_get_shared_fitness():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    IND_SEMANTICS1 = np.array([0, 1, 1, 0])
    IND_FITNESS1 = 0

    fsf = FitnessSharingFunction(X, y)

    assert fsf.get_shared_fitness(IND_SEMANTICS1, IND_FITNESS1) == np.ones(IND_SEMANTCS.size)
