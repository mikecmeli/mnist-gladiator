import mnist
from PIL import Image
import scipy.misc
import random

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()
im = Image.fromarray(train_images[0,:,:])
im.save('out.jpeg', "JPEG")