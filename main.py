# RGB - стандартный формат
# BGR - формат в OpenCV


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

# img = cv2.imread('/home/sam/Изображения/Tiger.jpg')
#
# img = cv2.resize(img, (img.shape[1], img.shape[0]))
# img = cv2.GaussianBlur(img, (9, 9), 0)  # размытие (можно ставить только нечётные числа)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img = cv2.Canny(img, 100, 100)
#
# kernel = np.ones(((5, 5)), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
#
# img = cv2.erode(img, kernel, iterations=1)
#
# cv2.imshow('Yes, its works!', img)
#
# print(img.shape)
#
# cv2.waitKey(0)

# ----------------------------------------------------------------------------------------------------------------------

#'/home/sam/Видео/Cat.mp4'
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#
#     # img = cv2.resize(img, (img.shape[1], img.shape[0]))
#     img = cv2.GaussianBlur(img, (9, 9), 0)  # размытие (можно ставить только нечётные числа)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.Canny(img, 30, 30) # чем меньше значение, тем больше точность
#
#     kernel = np.ones(((5, 5)), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#
#     img = cv2.erode(img, kernel, iterations=1)
#
#     cv2.imshow('Yes, its works!', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# ----------------------------------------------------------------------------------------------------------------------
# ПОИСК КОНТУРОВ И ПЕРЕНОС ИХ В НОВУЮ ПЕРЕМЕННУЮ
# img = cv2.imread('/home/sam/Изображения/Tiger.jpg')
#
# #img = cv2.flip(img, 0)   # отражение по осям
#
# def rotate(img_paramrtr, angle):
#     height, width = img_paramrtr.shape[:2]
#     point = (width // 2, height // 2)
#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(img_paramrtr, mat, (width, height))
#
# # img = rotate(img, 90)   # поворот на указанное число градусов
#
# def transform(img_Parametr, x, y):
#     mat = np.float32([[1, 0, x], [0, 1, y]])
#     return cv2.warpAffine(img_Parametr, mat, (img_Parametr.shape[1],img_Parametr.shape[0]))
#
# # img = transform(img, 30, 200)   # смещение на указанные координаты
#
# new_img= np.zeros(img.shape, dtype='uint8')
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5,5), 0)
#
# img = cv2.Canny(img, 100, 140)
#
# contur, ier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
# cv2.drawContours(new_img, contur, -1, (255, 0, 255), 1)
#
# cv2.imshow('Yes, its works!', new_img)
# cv2.waitKey(0)
#-----------------------------------------------------------------------------------------------------------------------
# ИГРА С ЦВЕТОВЫМИ ФОРМАТАМИ
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
#
# R, G, B = cv2.split(img)
#
# img = cv2.merge([B, G, R])
# ----------------------------------------------------------------------------------------------------------------------
# ПОБИТОВЫЕ ОПЕРАЦИИ К ИЗОБРАЖЕНИЯМ, МАСКИ
# photo = cv2.imread('/home/sam/Изображения/Tiger.jpg')
# img = np.zeros(photo.shape[:2], dtype='uint8')
#
# circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
# square = cv2.rectangle(img.copy(), (250,250), (1250,1250), 255, -1)
#
# img = cv2.bitwise_and(photo, photo, mask=square)
# # img = cv2.bitwise_or(circle, square)
# # img = cv2.bitwise_xor(circle, square)
# # img = cv2.bitwise_not(square)
#
# cv2.imshow('Yes, its works!', img)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------------------------------------------------
# НАХОЖДЕНИЕ ЛИЦ
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#
#     # img = cv2.resize(img, (img.shape[1], img.shape[0]))
#     img = cv2.GaussianBlur(img, (5, 5), 0)  # размытие (можно ставить только нечётные числа)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # img = cv2.Canny(img, 50, 50) # чем меньше значение, тем больше точность
#
#     # kernel = np.ones(((5, 5)), np.uint8)
#     # img = cv2.dilate(img, kernel, iterations=1)
#     #
#     # img = cv2.erode(img, kernel, iterations=1)
#
#     faces = cv2.CascadeClassifier('faces.xml')
#
#     result = faces.detectMultiScale(gray_img, scaleFactor=2, minNeighbors=3)
#
#     for (x, y, w, h) in result:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
#         # cv2.circle(img, (x + h // 2, y + w // 2), ((h // 3) + (w // 3)), (0, 255, 0), thickness=2)
#
#
#     cv2.imshow('Yes, its works!', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#-----------------------------------------------------------------------------------------------------------------------
# МОНО ИЗМЕРЕНИЕ РАССТОЯНИЯ С ИСПОЛЬЗОВАНИЕМ НЕЙРОСЕТИ (.onnx)
import cv2
import mediapipe as mp
import time


mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

path_model = "models/"

# Read Network
model_name = "model-small.onnx"; # MiDaS v2.1 Small

# Load the DNN model
model = cv2.dnn.readNet(path_model + model_name)

if (model.empty()):
    print("Could not load the neural net! - Check path")


# Set backend and target to CUDA to use GPU
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def depth_to_distance(depth):
    return -1.7 * depth + 2


cap = cv2.VideoCapture(0)

with mp_facedetector.FaceDetection(min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():

        success, img = cap.read()

        imgHeight, imgWidth, channels = img.shape

        start = time.time()

        # ----------------------------------------------------------------------------------

        # Convert the BGR image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --------- Process the image and find faces with mediapipe ---------
        results = face_detection.process(img)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(img, detection)
                # print(id, detection)

                bBox = detection.location_data.relative_bounding_box

                h, w, c = img.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # -------------- Depth map from neural net ---------------------------
        # Create Blob from Input Image

        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

        # Set input to the model
        model.setInput(blob)

        # Make forward pass in model
        depth_map = model.forward()

        depth_map = depth_map[0, :, :]
        depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

        # Normalize the output
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert the image color back so it can be displayed
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ----------------------------------------------------------------------------------------

        # Depth to face
        depth_face = depth_map[int(center_point[1]), int(center_point[0])]

        depth_face = depth_to_distance(depth_face)
        # print("Depth to face: ", depth_face)
        cv2.putText(img, "Depth in cm: " + str(round(depth_face, 2) * 100), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)

        # Depth converted to distance

        # ----------------------------------------------------------------------------------------
        end = time.time()
        totalTime = end - start

        fps = 5 / totalTime
        # print("FPS: ", fps)

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # cv2.imshow('Face Detection', img)
        cv2.imshow('Depth map', depth_map)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release(a)
# ----------------------------------------------------------------------------------------------------------------------
