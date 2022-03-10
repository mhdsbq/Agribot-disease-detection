# import open cv and numpy
import imp
import cv2
import numpy as np
from object_detector import object_detector
from load_model import load_model

cap = cv2.VideoCapture(0)
model = load_model()

while True:
    ret, frame = cap.read()
    frame = frame[::, 80:560]
    cv2.imshow('window name', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    # print(frame.shape)
    # (480, 640, 3)
    image = cv2.resize(frame, (256,256))
    print(image.shape)

    # cv2.imshow('2nd window', image)
    prediction = object_detector(model, image)
    print(prediction)
    

cap.release()
cv2.destroyAllWindows()

