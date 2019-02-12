import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/monarch.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

pixel_vals = image.reshape((-1, 3))

pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)
plt.show()
