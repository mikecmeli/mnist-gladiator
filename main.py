import mnist
from PIL import Image
from naive_bayes import naive_bayes
import numpy as np

pixel_threshold = 127

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

on_pixels = train_images > pixel_threshold

train_images_threshold = np.zeros_like(train_images)
train_images_threshold[on_pixels] = 1

correct_percent, guesses = naive_bayes(train_images_threshold, train_labels, train_images_threshold, train_labels)

print(correct_percent)

# im = Image.fromarray(np.reshape(train_images_threshold[0:100,:,:] * 255,(28*100,28)))
# im.save('out.jpeg', "JPEG")
# print(np.max(train_images[0,:,:]), np.min(train_images[0,:,:]))