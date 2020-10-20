import numpy as np
import utilities as uti
from cv2 import cv2
from datetime import datetime
from pyrealsense2 import pyrealsense2 as rs


# '''
# realsense version
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 480, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
now = datetime.now()
save_path = '/home/xihan/Videos/data_realsense_raw{}.avi'.format(now)
print(now)
out = cv2.VideoWriter(save_path, fourcc, 20.0, (480, 480))

isRecording = False  # flag to start recording

try:
    while(True):
        key = cv2.waitKey(33)
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        frame = color_image
        # frame = frame[:, 80:560]

        if isRecording:
            out.write(frame)

        cv2.imshow('frame', frame)

        if key == ord('q'):   # quit
            print('exit')
            break
        elif key == ord('s'):   # start recording
            isRecording = True
            print("start recording")
        elif key == ord('e'):   # end recording
            isRecording = False
            print("end recording")
finally:
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
# '''


'''
# webcam version
# Define the codec and create VideoWriter object
cap = cv2.VideoCapture(2)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
now = datetime.now()
save_path = '/home/xihan/Videos/data_webcam_raw{}.avi'.format(now)
print(now)
out = cv2.VideoWriter(save_path, fourcc, 20.0, (480, 480))

isRecording = False  # flag to start recording

while(True):
    key = cv2.waitKey(33)
    ret, frame = cap.read()
    frame = frame[:, 80:560]

    if isRecording:
        out.write(frame)

    cv2.imshow('frame', frame)

    if key == ord('q'):   # quit
        print('exit')
        break
    elif key == ord('s'):   # start recording
        isRecording = True
        print("start recording")
    elif key == ord('e'):   # end recording
        isRecording = False
        print("end recording")

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
'''
