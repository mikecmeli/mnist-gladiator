import scipy.misc
import random
import numpy as np

def naive_bayes(train_x,train_y,test_x,test_y):
    epsilon = 0.1
    num_test = 6000

    nb_matrix = np.zeros((10,train_x.shape[1],train_x.shape[2]))
    counts = np.zeros(10)
    for i in range(10):
        indices = np.nonzero(train_y == i)
        counts[i] = len(indices[0])
        sum = np.sum(train_x[indices,:,:], axis=(0,1))
        if(counts[i] == 0):
            counts[i] = 1
        nb_matrix[i,:,:] = sum/counts[i]

    print(np.nonzero(nb_matrix > 1))
    print(np.nonzero(nb_matrix < 0))

    guesses = []

    # need to vectorize
    for i in range(num_test): # need to change this
        on_pixels = np.nonzero(test_x[i] == 1)
        off_pixels = np.nonzero(test_x[i] == 0)
        # print(len(on_pixels[0]))
        # print(len(off_pixels[0]))
        probs = np.zeros((10,test_x.shape[1],test_x.shape[2]))
        for j in range(10): # need to change this
            probs[j,on_pixels[0],on_pixels[1]] = nb_matrix[j,on_pixels[0],on_pixels[1]] + epsilon
            probs[j,off_pixels[0],off_pixels[1]] = 1 - nb_matrix[j,off_pixels[0],off_pixels[1]] + epsilon
        # print(np.nonzero(probs < 0))

        # print(probs[0])
        guesses.append(np.argmax(np.prod(probs, axis=(1,2)))) # product here will probably underflow
        # print(np.prod(probs, axis=(1,2)))

    guesses = np.asarray(guesses)


    print(guesses)


    equal_indices = test_y[0:num_test] == guesses
    equal_amount = equal_indices.sum()
    return(equal_amount/num_test, guesses)
