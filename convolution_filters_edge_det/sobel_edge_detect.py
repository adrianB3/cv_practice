import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

image = mpimg.imread('images/curved_lane.jpg')
plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

img_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

filtered_img_x = cv2.filter2D(img_blurred, -1, sobel_x)
filtered_img_y = cv2.filter2D(img_blurred, -1, sobel_y)

filtered_img = filtered_img_x + filtered_img_y
retval, filtered_img_thresh = cv2.threshold(filtered_img, 100, 255, cv2.THRESH_BINARY)

plt.imshow(filtered_img_thresh, cmap='gray')
plt.show()
