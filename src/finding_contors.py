# FINDING CONTOURS AND TRANSFERRING THEM TO A NEW VARIABLE

import cv2
import numpy as np

img = cv2.imread('<the path to the image>')

#img = cv2.flip(img, 0)   # reflection on the axes

def rotate(img_paramrtr, angle):
    height, width = img_paramrtr.shape[:2]
    point = (width // 2, height // 2)
    mat = cv2.getRotationMatrix2D(point, angle, 1)
    return cv2.warpAffine(img_paramrtr, mat, (width, height))

# img = rotate(img, 90)   # rotate by the specified number of degrees

def transform(img_Parametr, x, y):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img_Parametr, mat, (img_Parametr.shape[1],img_Parametr.shape[0]))

# img = transform(img, 30, 200)   # offset by the specified coordinates

new_img= np.zeros(img.shape, dtype='uint8')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 0)

img = cv2.Canny(img, 100, 140)

contur, ier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(new_img, contur, -1, (255, 0, 255), 1)

cv2.imshow('Yes, its works!', new_img)
cv2.waitKey(0)
