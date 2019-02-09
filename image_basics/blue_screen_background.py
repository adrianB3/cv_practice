import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/pizza_bluescreen.jpg')

print('This image is ', type(image), "With dimns: ", image.shape)

img_cpy = np.copy(image)
img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)

plt.imshow(img_cpy)

# color thresholds
lower_blue = np.array([0, 0, 225])
upper_blue = np.array([225, 225, 255])

# create mask
mask = cv2.inRange(img_cpy, lower_blue, upper_blue)
plt.imshow(mask, cmap='gray')

masked_image = np.copy(img_cpy)

masked_image[mask != 0] = [0, 0, 0]
plt.imshow(masked_image)

back_img = cv2.imread('images/space_background.jpg')
back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)

cropped_back = back_img[0:514, 0:816]
cropped_back[mask == 0] = [0, 0, 0]

plt.imshow(cropped_back)

# create final image
complete_img = masked_image + cropped_back

plt.imshow(complete_img)
