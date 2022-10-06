# PLAYING WITH COLOR FORMATS
import cv2

img = cv2.imread('<the path to the image>')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

R, G, B = cv2.split(img)

img = cv2.merge([B, G, R])
