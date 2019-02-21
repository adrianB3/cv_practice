import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

print (tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data\\MNIST\\')

print("Size of: ")
print(" - Training-set:\t\t{}".format(data.train))
print(" - Validation-set:\t{}".format(data.num_val))
print(" - Test-set:\t\t{}".format(data.num_test))

img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes
# TODO data import not working