import cv2

img = cv2.imread('images/sky.jpg')
cv2.startWindowThread()
cv2.namedWindow("Hello")
cv2.imshow("Hello", img)
cv2.waitKey()