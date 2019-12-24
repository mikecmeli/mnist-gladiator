import mnist
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

on_pixels = test_images > pixel_threshold

test_images_threshold = np.zeros_like(test_images)
test_images_threshold[on_pixels] = 1

correct_percent, guesses = naive_bayes(
    train_images_threshold, train_labels, test_images_threshold, test_labels
)

print(correct_percent)
