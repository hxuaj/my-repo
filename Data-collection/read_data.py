# -*- coding: UTF-8 -*-

import cv2

capture = cv2.VideoCapture('Data\output.avi')

#Get fps in real life.
fps = capture.get(5)

i = 0

while(capture.isOpened()):

    ret, frame = capture.read()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    i += 1
    if i == 10:
        cv2.imwrite('test.png', frame)

    keypress = cv2.waitKey(int((1000 / fps)))
    if keypress & 0xFF == ord('q'):
        break
    else:
            print("The resolution is {} x {}".format(capture.get(3), capture.get(4)))
            print("The frame rate: {}".format(capture.get(5)))

cap.release()
cv2.destroyAllWindows()