# -*- coding: UTF-8 -*-

import cv2

capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Data\output.avi',fourcc, 20.0, (640,480))

while(capture.isOpened()):
    # Get one frame
    ret, frame = capture.read()

    # Change image into gray level
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret is True:
        # frame = cv2.flip(frame, 0)
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            # print("The resolution is {} x {}".format(capture.get(3), capture.get(4)))
            # print("The frame rate: {}".format(capture.get(7)))
            pass
    else:
        break

capture.release()
out.release()
cv2.destroyAllWindows()