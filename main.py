import mnist
from PIL import Image
import scipy.misc
import random
import numpy as np

epsilon = 0.1
pixel_threshold = 127
num_test = 200

train_images = mnist.train_images()[0:num_test]
train_labels = mnist.train_labels()[0:num_test]

test_images = mnist.test_images()
test_labels = mnist.test_labels()

on_pixels = train_images > pixel_threshold

train_images_threshold = np.zeros_like(train_images)
train_images_threshold[on_pixels] = 1

# im = Image.fromarray(np.reshape(train_images_threshold[0:100,:,:] * 255,(28*100,28)))
# im.save('out.jpeg', "JPEG")
# print(np.max(train_images[0,:,:]), np.min(train_images[0,:,:]))
nb_matrix = np.zeros((10,train_images.shape[1],train_images.shape[2]))
counts = np.zeros(10)
for i in range(10):
    indices = np.nonzero(train_labels == i)
    counts[i] = len(indices)
    sum = np.sum(train_images_threshold[indices,:,:], axis=(0,1))
    if(counts[i] == 0):
        counts[i] = 1
    nb_matrix[i,:,:] = sum/counts[i]

# print(nb_matrix[0])

x_list = []

# need to vectorize
for i in range(num_test): # need to change this
    on_pixels = np.nonzero(train_images_threshold[i] == 1)
    off_pixels = np.nonzero(train_images_threshold[i] == 0)
    print(len(on_pixels[0]))
    print(len(off_pixels[0]))
    probs = np.zeros((10,train_images.shape[1],train_images.shape[2]))
    for j in range(10): # need to change this
        probs[j,on_pixels[0],on_pixels[1]] = nb_matrix[j,on_pixels[0],on_pixels[1]] + epsilon
        probs[j,off_pixels[0],off_pixels[1]] = 1 - nb_matrix[j,off_pixels[0],off_pixels[1]] + epsilon

    print(probs[0])
    x_list.append(np.argmax(np.prod(probs, axis=(1,2)))) # product here will probably underflow
    print(np.prod(probs, axis=(1,2)))

x_list = np.asarray(x_list)

print(train_labels[0:num_test])
print(x_list)

equal_indices = train_labels[0:num_test] == x_list
print(equal_indices)
equal_amount = equal_indices.sum()
print(equal_amount/num_test)
