import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/waffle.jpg')
img_cpy = np.copy(image)
img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
plt.imshow(img_cpy)
plt.show()

gray = cv2.cvtColor(img_cpy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

plt.imshow(dst, cmap='gray')

thresh = 0.1 * dst.max()

corner_image = np.copy(img_cpy)

for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if(dst[j, i] > thresh):
            cv2.circle(corner_image, (i, j), 1, (0, 255, 0), 1)

plt.imshow(corner_image)
plt.show()
