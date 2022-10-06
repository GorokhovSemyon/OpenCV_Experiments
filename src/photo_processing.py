# PHOTO PROCESSING
# RGB - standard format
# BGR - OpenCV format

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# photo = np.zeros((450, 450, 3), dtype='uint8')
#
# # photo[100:150, 200:300] = 26, 240, 250
#
# cv2.rectangle(photo, (photo.shape[0] // 4, photo.shape[1] // 4), (photo.shape[0] // 4 * 3,photo.shape[0] // 4 * 3), (26, 240, 250), thickness=2)
# cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[0] // 2), (26, 240, 250), thickness=2)
# cv2.line(photo, (0, 0), (photo.shape[0], photo.shape[0]), (26, 240, 250), thickness=2)
# cv2.line(photo, (0, photo.shape[0]), (photo.shape[0], 0), (26, 240, 250), thickness=2)
# cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), photo.shape[0] // 8, (26, 240, 250), thickness=cv2.FILLED)
# cv2.putText(photo, 'WooooW', (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 3)
#
# cv2.imshow('Creation', photo)
# cv2.waitKey(0)

# ----------------------------------------------------------------------------------------------------------------------

img = cv2.imread('<the path to the image>')

img = cv2.resize(img, (img.shape[1], img.shape[0]))
img = cv2.GaussianBlur(img, (9, 9), 0)  # blur (you can only put odd numbers)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.Canny(img, 100, 100)

kernel = np.ones(((5, 5)), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

img = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Yes, its works!', img)

print(img.shape)

cv2.waitKey(0)
