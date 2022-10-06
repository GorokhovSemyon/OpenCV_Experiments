# VIDEO PROCESSING

import cv2
import numpy as np

'<the path to the video>'
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # img = cv2.resize(img, (img.shape[1], img.shape[0]))
    img = cv2.GaussianBlur(img, (9, 9), 0)  # blur (you can only put odd numbers)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.Canny(img, 30, 30) # the smaller the value, the greater the accuracy

    kernel = np.ones(((5, 5)), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.erode(img, kernel, iterations=1)

    cv2.imshow('Yes, its works!', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break