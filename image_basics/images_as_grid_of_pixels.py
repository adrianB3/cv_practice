import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

image = mpimg.imread('images/waymo_car.jpg')

print('Image dimensions: ', image.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')
plt.show()
cv2.imshow("Hello", gray_image)
cv2.waitKey(0)