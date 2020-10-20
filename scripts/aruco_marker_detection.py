import numpy as np
from cv2 import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

# frame = cv2.imread("../test_images/markers6.png")
frame = cv2.imread("../test_images/markers4.png")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(
    gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

print(corners)

plt.figure()
plt.imshow(frame_markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:, 0].mean()], [c[:, 1].mean()],
             "o", label="id={0}".format(ids[i]))
# plt.legend()
plt.show()
