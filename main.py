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
    sum = np.sum(train_images[indices,:,:], axis=(0,1))
    nb_matrix[i,:,:] = sum/counts[i]

x_list = []

# need to vectorize
for i in range(1000): # need to change this
    on_pixels = train_images_threshold[i] == 1
    off_pixels = train_images_threshold[i] == 0
    probs = np.zeros((10,train_images.shape[1],train_images.shape[2]))
    for j in range(10): # need to change this
        probs[j,on_pixels] = nb_matrix[j,on_pixels] + epsilon
        probs[j,off_pixels] = 1 - nb_matrix[j,off_pixels] + epsilon
    
    x_list.append(np.argmax(np.prod(probs, axis=(0,1)))) # product here will probably underflow

x_list = np.asarray(x_list)

print(train_labels[0:1000])
print(x_list)

equal_indices = train_labels[0:1000] == x_list
print(equal_indices)
equal_amount = equal_indices.sum()
print(equal_amount/1000)
