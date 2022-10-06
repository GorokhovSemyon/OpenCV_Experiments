# FINDING FACES

import cv2
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # img = cv2.resize(img, (img.shape[1], img.shape[0]))
    img = cv2.GaussianBlur(img, (5, 5), 0)  # blur (you can only put odd numbers)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.Canny(img, 50, 50) # the smaller the value, the greater the accuracy

    # kernel = np.ones(((5, 5)), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    #
    # img = cv2.erode(img, kernel, iterations=1)

    faces = cv2.CascadeClassifier('faces.xml')

    result = faces.detectMultiScale(gray_img, scaleFactor=2, minNeighbors=3)

    for (x, y, w, h) in result:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        # cv2.circle(img, (x + h // 2, y + w // 2), ((h // 3) + (w // 3)), (0, 255, 0), thickness=2)


    cv2.imshow('Yes, its works!', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
