# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/brain_MR.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 200, 240)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')
ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
