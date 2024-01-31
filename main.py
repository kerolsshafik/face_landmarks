import os
import cv2
import dlib
import matplotlib.pyplot as plt
# from keras.models import Model, load_model
# from keras.models import Model, Sequential
# from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from cv2 import imread

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cap = cv2


img_counter = 0
# cv2.namedWindow("test")

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)

names= []
# images path
path = "C:/Users/kerols shafik/PycharmProjects/fave_landmarks/data/normal"
for name in os.listdir(path):
    print(name)
    names.append(name)


c=0
for filename in os.listdir(path):
    # ret, image = cap.read()
    image=cv2.imread(os.path.join(path,names[c]))

    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # Display the image
    # cv2.imshow('Landmark Detection', image)
    lol="C:/Users/kerols shafik/PycharmProjects/fave_landmarks/data/landmarkslie"

    cv2.imwrite("landmarks_lie/"+ names[c], image)


    # k = cv2.waitKey(1)
    # # Press the escape button to terminate the code
    # if k & 0xFF == ord('q'):
    #     break

    # elif k % 256 == 32:
    #     # SPACE pressed
    #     img_name = "C:/Users/kerols shafik/PycharmProjects/fave_landmarks/data/lie/opencv_frame0_{}.png".format(img_counter)
    #     cv2.imwrite(img_name, image)
    #     print("{} written!".format(img_name))
    #     img_counter += 1
    c += 1

# cap.release()
# cv2.destroyAllWindows()