import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/multi_faces.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 10))
plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(20, 10))
plt.imshow(gray, cmap='gray')
plt.show()

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 4, 6)

print("Found: " + str(len(faces)) + " faces.")
print(faces)

img_with_detections = np.copy(image)

for (x, y, w, h) in faces:
    cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 5)

plt.figure(figsize=(20, 10))
plt.imshow(img_with_detections)
plt.show()
