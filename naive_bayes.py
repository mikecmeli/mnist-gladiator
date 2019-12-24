import random
import numpy as np


def naive_bayes(train_x, train_y, test_x, test_y, **kwargs):
    epsilon = 0.001 if "epsilon" not in kwargs else kwargs["epsilon"]

    nb_matrix = np.zeros((10, train_x.shape[1], train_x.shape[2]))
    counts = np.zeros(10)
    for i in range(10):
        indices = np.nonzero(train_y == i)
        counts[i] = len(indices[0])
        sum = np.sum(train_x[indices, :, :], axis=(0, 1))
        if counts[i] == 0:
            counts[i] = 1
        nb_matrix[i, :, :] = sum / counts[i]

    guesses = []

    # need to vectorize
    for i in range(len(test_y)):  # need to change this
        on_pixels = np.nonzero(test_x[i] == 1)
        off_pixels = np.nonzero(test_x[i] == 0)
        probs = np.zeros((10, test_x.shape[1], test_x.shape[2]))
        for j in range(10):  # need to change this
            probs[j, on_pixels[0], on_pixels[1]] = (
                nb_matrix[j, on_pixels[0], on_pixels[1]] + epsilon
            )
            probs[j, off_pixels[0], off_pixels[1]] = (
                1 - nb_matrix[j, off_pixels[0], off_pixels[1]] + epsilon
            )

        guesses.append(
            np.argmax(np.prod(probs, axis=(1, 2)))
        )  # product here will probably underflow

    guesses = np.asarray(guesses)

    equal_indices = test_y == guesses
    equal_amount = equal_indices.sum()
    return (equal_amount / len(test_y), guesses)
