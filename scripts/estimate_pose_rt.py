'''
estimate fiducial marker 6dof pose
'''

import numpy as np
from cv2 import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import math


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


cap = cv2.VideoCapture(2)
focus = 0               # min: 0, max: 255, increment:5
cap.set(28, focus)      # manually set focus

# rotx90 = [[1, 0, 0], [
#     0, np.cos(3.14/2), -np.sin(3.14/2)], [0, np.sin(3.14/2), np.cos(3.14/2)]]

camera_matrix = np.array(
    [[662.1790, 0.0, 322.3619], [0.0, 662.8344, 252.0131], [0.0, 0.0, 1.0]])
dist_coeff = np.array([0.0430651, -0.1456001, 0.0, 0.0])

angle_ind = 0  # marker id
font = cv2.FONT_HERSHEY_SIMPLEX

time = list()
rotx_rec = list()
roty_rec = list()
rotz_rec = list()
curr_time = 0

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    # pose estimation
    try:
        loc_marker = corners[np.where(ids == angle_ind)[0][0]]
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
            loc_marker, 0.048, camera_matrix, dist_coeff)
        axis_frame = aruco.drawAxis(
            frame, camera_matrix, dist_coeff, rvecs, tvecs, 0.15)

        rmat = cv2.Rodrigues(rvecs)[0]
        # rmat = np.matmul(rmat, rotx90)
        RotX = rotationMatrixToEulerAngles(rmat)[1]
        RotX_formatted = float("{0:.2f}".format(-RotX*180/3.14))     # 2 digits
        RotY = rotationMatrixToEulerAngles(rmat)[0]
        if RotY > 0:
            RotY = 3.14 - RotY
        elif RotY <= 0:
            RotY = -3.14 - RotY
        RotY_formatted = float("{0:.2f}".format(RotY*180/3.14))
        RotZ = rotationMatrixToEulerAngles(rmat)[2]
        RotZ_formatted = float("{0:.2f}".format(RotZ*180/3.14))

        cv2.putText(axis_frame, 'rotX [deg]:',
                    (5, 40), font, 0.8, (1, 100, 1), thickness=2)
        cv2.putText(axis_frame, str(RotX_formatted),
                    (150, 40), font, 0.8, (1, 100, 1), thickness=2)
        cv2.putText(axis_frame, 'rotY [deg]:',
                    (5, 80), font, 0.8, (1, 100, 1), thickness=2)
        cv2.putText(axis_frame, str(RotY_formatted),
                    (150, 80), font, 0.8, (1, 100, 1), thickness=2)
        cv2.putText(axis_frame, 'rotZ [deg]:',
                    (5, 120), font, 0.8, (1, 100, 1), thickness=2)
        cv2.putText(axis_frame, str(RotZ_formatted),
                    (150, 120), font, 0.8, (1, 100, 1), thickness=2)

        time.append(curr_time)
        rotx_rec.append(RotX_formatted)
        roty_rec.append(RotY_formatted)
        rotz_rec.append(RotZ_formatted)

        cv2.imshow('frame', axis_frame)

    except:
        cv2.imshow('frame', frame_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # quit
        print('exit')
        break

    curr_time = curr_time + 1


# fig = plt.figure()
# plt.plot(time, rotx_rec, label='rotation about X')
# plt.hold(True)
# plt.plot(time, roty_rec, label='rotation about Y')
# plt.ylabel('degree')
# plt.xlabel('time stamp')
# plt.legend(loc="upper left")
# plt.show()

cap.release()
cv2.destroyAllWindows()
