# BITWISE OPERATIONS TO IMAGES, MASKS

import cv2
import numpy as np

photo = cv2.imread('<the path to the image>')
img = np.zeros(photo.shape[:2], dtype='uint8')

circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
square = cv2.rectangle(img.copy(), (250,250), (1250,1250), 255, -1)

img = cv2.bitwise_and(photo, photo, mask=square)
# img = cv2.bitwise_or(circle, square)
# img = cv2.bitwise_xor(circle, square)
# img = cv2.bitwise_not(square)

cv2.imshow('Yes, its works!', img)
cv2.waitKey(0)
