import cv2
import numpy as np
car_cas = cv2.CascadeClassifier('haarcascade_car.xml')

video = cv2.VideoCapture('car_mov.mp4')


while video.isOpened():
    _,frame = video.read()
    if np.shape(frame) == ():
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    car = car_cas.detectMultiScale(gray, 1.6, 2)
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
video.release()
cv2.destroyAllWindows()