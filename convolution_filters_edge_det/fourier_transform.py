# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_stripes = cv2.imread('images/stripes.jpg')
image_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_BGR2RGB)

image_solid = cv2.imread('images/pink_solid.jpg')
image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplot(1, 2, figsize=(10, 5))
ax1.imshow(image_stripes)
ax2.imshow(image_solid)
