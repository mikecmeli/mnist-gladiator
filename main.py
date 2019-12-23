import mnist
from PIL import Image
import scipy.misc
import random
import numpy as np

epsilon = 0.01
pixel_threshold = 127

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()



im = Image.fromarray(np.reshape(train_images[0:100,:,:],(28*100,28)))
im.save('out.jpeg', "JPEG")
print(np.max(train_images[0,:,:]), np.min(train_images[0,:,:]))
nb_matrix = np.zeros((10,train_images.shape[1],train_images.shape[2]))
counts = np.zeros(10)
for i in range(10):
    indices = np.nonzero(train_labels == i)
    counts[i] = len(indices)
    sum = np.sum(train_images[indices,:,:], axis=(0,1))
    nb_matrix[i,:,:] = sum/counts[i]



