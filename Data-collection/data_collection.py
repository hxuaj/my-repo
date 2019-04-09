# -*- coding: UTF-8 -*-

import cv2

# Start the default camera
capture = cv2.VideoCapture(0)
default_camera_width = int(capture.get(3))
default_camera_height = int(capture.get(4))
print("The camera's default resolution is {} x {}".format(default_camera_width, default_camera_height))

# Set capture FPS
camera_fps = 20
capture_fps = 20
# Save a pic every i seconds
i = 5
count_frame = 0
pic_index = 0

# Record for rec_time seconds
rec_time = 30

# Define the codec and create VideoWriter object
resolution_scale = 1/2

capture_width = int(default_camera_width * resolution_scale)
capture_height = int(default_camera_height * resolution_scale)

capture.set(3, capture_width)
capture.set(4, capture_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Data-video\output.avi',fourcc, capture_fps, (capture_width,capture_height))

while(capture.isOpened()):
    # Get one frame
    ret, frame = capture.read()

    # Change image into gray level
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret is True:
        # frame = cv2.flip(frame, 0)

        # Save video
        out.write(frame)
        # Save pic per i senconds
        count_frame += 1
        if count_frame == i * camera_fps:
            count_frame = 0
            cv2.imwrite("Data-pic\pic" + str(pic_index) + ".jpg", frame)
            pic_index += 1
            # break if the time reaches rec_time
            if (pic_index + 1) * i > rec_time:
                break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            # print("The resolution is {} x {}".format(capture.get(3), capture.get(4)))
            # print("The frame rate: {}".format(capture.get(5)))
            pass
    else:
        break

capture.release()
out.release()
cv2.destroyAllWindows()