# -*- coding: UTF-8 -*-

import cv2
from optparse import OptionParser


usage = "data_collection.py[ -f <capture_fps>][-i <pic_save_freq>][-r <rec_time>][-rs <resolution_scale>]"
opt = OptionParser(usage)
opt.add_option("-f","--F",action = "store",type="int",dest = "capture_fps", 
                default = 20, help = "Set the fps for video capturing.")
opt.add_option("-i","--I",action = "store",type="int",dest = "pic_save_freq", 
                default = 5, help = "Set the freq to save pictures.")
opt.add_option("-r","--R",action = "store",type="int",dest = "rec_time", 
                default = 30, help = "Record for rec_time seconds.")
opt.add_option("-s","--S",action = "store",type="float",dest = "resolution_scale", 
                default = 1.0, help = "Define the resolution_scale w.r.t. VGA.")

options, args = opt.parse_args()

# Start the default camera
capture = cv2.VideoCapture(0)
default_camera_width = int(capture.get(3))
default_camera_height = int(capture.get(4))
print("The camera's default resolution is {} x {}".format(default_camera_width, default_camera_height))

# Set capture FPS
camera_fps = 20
if not options.capture_fps:
    capture_fps = camera_fps
else:
    capture_fps = options.capture_fps

# Save a pic every i seconds
if not options.pic_save_freq:
    i = 5
else:
    i = options.pic_save_freq

count_frame = 0
pic_index = 0

# Record for rec_time seconds
if not options.rec_time:
    rec_time = 30
else:
    rec_time = options.rec_time

# Define the codec and create VideoWriter object
if not options.resolution_scale:
    resolution_scale = 1
else:
    resolution_scale = options.resolution_scale

print("capture_fps = {}, pic_save_freq = {}, rec_time = {}, resolution_scale = {}".format(capture_fps, 
                                                                                          i,
                                                                                          rec_time,
                                                                                          resolution_scale))
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