import cv2
import helpers
import numpy as np
import matplotlib.pyplot as plt
import random


def standardize_input(image):
    standard_im = []
    standard_im = cv2.resize(image, (1100, 600))
    return standard_im


def encode(label):
    numerical_val = 0
    if label == "day":
        numerical_val = 1
    return numerical_val


def standardize(image_list):
    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]
        std_img = standardize_input(image)
        binary_label = encode(label)

        standard_list.append((std_img, binary_label))

    return standard_list


def avg_brightness(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    avg = 0
    sum_brightness = np.sum(hsv[:, :, 2])
    avg = sum_brightness / (rgb_image.shape[0] * rgb_image.shape[1])

    return avg


def estimate_label(rgb_image):
    predicted_label = 0
    avg = avg_brightness(rgb_image)
    threshold = 99
    if avg >= threshold:
        predicted_label = 1
    # TODO - find other features for day/night classification
    return predicted_label


def get_misclassified_images(test_images):
    misclassified_images_labels = []

    for image in test_images:
        im = image[0]
        true_label = image[1]

        predicted_label = estimate_label(im)

        if(predicted_label != true_label):
            misclassified_images_labels.append(
                (im, predicted_label, true_label))

    return misclassified_images_labels


image_dir_trainig = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# loading the training data
IMAGE_LIST = helpers.load_dataset(image_dir_trainig)
IMAGE_LIST_TEST = helpers.load_dataset(image_dir_test)

# visualizing an input image
image_index = 55
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

print("Image dimensions: ", selected_image.shape)
print("Label: ", selected_label)
plt.imshow(selected_image)
plt.show()

# a night image
image_index = 150
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]
print("Image dimensions: ", selected_image.shape)
print("Label: ", selected_label)
plt.imshow(selected_image)
plt.show()

STANDARDIZED_LIST = standardize(IMAGE_LIST)
STANDARDIZED_TEST_LIST = standardize(IMAGE_LIST_TEST)

image_index = 170
selected_image = STANDARDIZED_LIST[image_index][0]
selected_label = STANDARDIZED_LIST[image_index][1]

print("Image dimensions: ", selected_image.shape)
print("Label: ", selected_label)
plt.imshow(selected_image)
plt.show()
print("Image avg brightness: ", avg_brightness(selected_image))
est_label = estimate_label(selected_image)
print("Estimated label: ", est_label)

random.shuffle(STANDARDIZED_TEST_LIST)

MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# calculate accuracy
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print("Accuracy: ", accuracy)
print("Number of misclassified images: " +
      str(len(MISCLASSIFIED)) + " out of " + str(total))
