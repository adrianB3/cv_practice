import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/phone.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
# edge detection
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

low_thresh = 50
high_thresh = 100
edges = cv2.Canny(gray, low_thresh, high_thresh)

plt.imshow(edges, cmap='gray')
plt.show()
# hough params
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 110
max_line_gap = 5

line_image = np.copy(image)

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

plt.imshow(line_image)
plt.show()
