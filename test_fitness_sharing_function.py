from fitness_sharing_function import FitnessSharingFunction


def test_init():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y = [(0,), (1,), (1,), (0,)]

    fsf = FitnessSharingFunction(X, y)
