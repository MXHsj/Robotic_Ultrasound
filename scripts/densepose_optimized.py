# densepose rt with SVR correction
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
from joblib import dump, load
import math
import csv
from cv2 import cv2

camera_matrix = np.array(
    [[662.1790, 0.0, 322.3619], [0.0, 662.8344, 252.0131], [0.0, 0.0, 1.0]])

dist_coeff = np.array([0.0430651, -0.1456001, 0.0, 0.0])


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


def getBodyPart(IUV, part_id):
    IUV_chest = np.zeros((IUV.shape[0], IUV.shape[1], IUV.shape[2]))
    torso_idx = np.where(IUV[:, :, 0] == part_id)
    IUV_chest[torso_idx] = IUV[torso_idx]

    return IUV_chest


def divide2region(frame, IUV, IUV_chest, target_u, target_v, tip_coord):
    # mask = np.zeros((IUV.shape[0], IUV.shape[1], IUV.shape[2]))
    for reg in range(1, len(target_u)+1):
        u2xy_pair = np.where(
            IUV_chest[:, :, 1] == target_u[reg-1])    # xy paris in u
        v2xy_pair = np.where(
            IUV_chest[:, :, 2] == target_v[reg-1])    # xy pairs in v

        rcand = list()
        ccand = list()

        u_x = u2xy_pair[1]
        u_y = u2xy_pair[0]
        v_x = v2xy_pair[1]
        v_y = v2xy_pair[0]

        # need further optimization
        x_intersects = [x for x in u_x if x in v_x]
        y_intersects = [y for y in u_y if y in v_y]

        rcand = y_intersects
        ccand = x_intersects

        # u2xy_new = np.array(u2xy_pair).transpose()
        # v2xy_new = np.array(v2xy_pair).transpose()
        # try:
        #     xy_intersects = multidim_intersect(v2xy_new, u2xy_new)
        #     print(xy_intersects)
        #     # rcand.append(xy_intersects[1][0])
        #     # ccand.append(xy_intersects[0][0])
        #     # print(ccand)
        # except Exception as e:
        #     print('error: '+str(e))

        # for uind in range(len(u2xy_pair[0])):
        #     for vind in range(len(v2xy_pair[0])):
        #         x_u = u2xy_pair[1][uind]
        #         y_u = u2xy_pair[0][uind]
        #         x_v = v2xy_pair[1][vind]
        #         y_v = v2xy_pair[0][vind]
        #         if x_u == x_v and y_u == y_v:       # if xy pair intersects
        #             rcand.append(y_u)
        #             ccand.append(x_u)

        # print("\n rcand:", rcand, "\n ccand:", ccand)

        if len(rcand) > 0 and len(ccand) > 0:
            cen_col = int(np.mean(ccand))  # averaging col indicies
            cen_row = int(np.mean(rcand))  # averaging row indicies
            reg_coord = (cen_col, cen_row)
            frame = drawOverlay(frame, reg, reg_coord, tip_coord)

    return frame


def drawOverlay(frame, reg, reg_coord,  tip_coord):
    Radius = 15    # radius to mark circular region
    margin = 0
    default_color = (200, 200, 200)     # white
    scan_color = (80, 50, 50)
    text_color = (1, 100, 1)    # green
    font = cv2.FONT_HERSHEY_SIMPLEX
    if tip_coord[0] != -1 and tip_coord[1] != -1:
        dist = np.sqrt((reg_coord[0]-tip_coord[0])*(reg_coord[0]-tip_coord[0]) +
                       (reg_coord[1]-tip_coord[1])*(reg_coord[1]-tip_coord[1]))
        if dist <= Radius+margin:
            color = scan_color
        else:
            color = default_color
    else:
        color = default_color
    frame = cv2.circle(frame, reg_coord, Radius, color, -1)
    cv2.putText(frame, str(reg), reg_coord, font, 0.5, text_color, thickness=2)

    return frame


def detectFiducial(frame):
    # marker detection
    fid_id = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # pose estimation
    try:
        loc_marker = corners[np.where(ids == fid_id)[0][0]]
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
            loc_marker, 0.05, camera_matrix, dist_coeff)
        marker_frame = aruco.drawAxis(
            frame, camera_matrix, dist_coeff, rvecs, tvecs, 0.15)

        rmat = cv2.Rodrigues(rvecs)[0]
        RotX = rotationMatrixToEulerAngles(rmat)[1]
        RotX_formatted = float("{0:.2f}".format(-RotX*180/3.14))     # 2 digits
    except:
        RotX_formatted = -1

    return marker_frame, RotX_formatted


def detectMarker(frame):
    # marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # marker position
    num_markers = 4
    pos = np.zeros((num_markers, 2))
    for i in range(1, num_markers+1):
        try:
            marker = corners[np.where(ids == i)[0][0]][0]
            pos[i-1, :] = [marker[:, 0].mean(), marker[:, 1].mean()]
        except:
            pos[i-1, :] = [-1, -1]      # if marker is not detected
        # print("id{} center:".format(i), pos[i-1, 0], pos[i-1, 1])

    return marker_frame, pos


def getUV(IUV_chest, pos):
    UV = np.zeros((pos.shape[0], pos.shape[1]))
    for id in range(pos.shape[0]):
        if pos[id, 0] != -1 and pos[id, 1] != -1:
            row = int(pos[id, 1])
            col = int(pos[id, 0])
            UV[id, 0] = IUV_chest[row, col, 1]     # store U
            UV[id, 1] = IUV_chest[row, col, 2]     # store V
        else:
            UV[id, 0] = -1
            UV[id, 1] = -1
    return UV


def initVideoStream():
    cap = cv2.VideoCapture(2)
    focus = 0               # min: 0, max: 255, increment:5
    cap.set(28, focus)      # manually set focus
    return cap


def getVideoStream(cap):
    patch_size = 480
    _, frame = cap.read()
    frame = frame[:, 80:560]
    return frame


def main():
    part_id = 2
    save_path = '/home/xihan/Myworkspace/lung_ultrasound/image_buffer/incoming.png'
    tip_x = -1  # tip tracking is not necessary
    tip_y = -1
    # initial UV values
    target_u = [60, 100, 60, 100]
    target_v = [152, 167, 85, 82]
    # load trained models
    svr_1u_load = load('trained_models/marker1u.joblib')
    svr_1v_load = load('trained_models/marker1v.joblib')
    svr_2u_load = load('trained_models/marker2u.joblib')
    svr_2v_load = load('trained_models/marker2v.joblib')
    svr_5u_load = load('trained_models/marker5u.joblib')
    svr_5v_load = load('trained_models/marker5v.joblib')
    svr_6u_load = load('trained_models/marker6u.joblib')
    svr_6v_load = load('trained_models/marker6v.joblib')

    cap = initVideoStream()

    while(True):
        frame = getVideoStream(cap)
        key = cv2.waitKey(33)
        cv2.imwrite(save_path, frame)
        frame, angleX = detectFiducial(frame)   # detect fiducial marker
        frame, pos = detectMarker(frame)        # detect overlay marker

        # predict UV
        if angleX != -1:
            y_1u_pred = int(svr_1u_load.predict([[angleX]]))
            y_1v_pred = int(svr_1v_load.predict([[angleX]]))
            y_2u_pred = int(svr_2u_load.predict([[angleX]]))
            y_2v_pred = int(svr_2v_load.predict([[angleX]]))
            y_5u_pred = int(svr_5u_load.predict([[angleX]]))
            y_5v_pred = int(svr_5v_load.predict([[angleX]]))
            y_6u_pred = int(svr_6u_load.predict([[angleX]]))
            y_6v_pred = int(svr_6v_load.predict([[angleX]]))
            target_u = [y_1u_pred, y_2u_pred, y_5u_pred, y_6u_pred]
            target_v = [y_1v_pred, y_2v_pred, y_5v_pred, y_6v_pred]
            print("target u: ", target_u)
            print("target V: ", target_v)

        try:
            inferred = cv2.imread(
                '/home/xihan/Myworkspace/lung_ultrasound/infer_out/incoming_IUV.png')
        except Exception as e:
            print('error: '+str(e))

        if inferred is not None:
            IUV_chest = getBodyPart(inferred, part_id)
            frame = divide2region(frame, inferred, IUV_chest,
                                  target_u, target_v, (tip_x, tip_y))
            UV = getUV(IUV_chest, pos)
        else:
            UV = -1*np.ones((4, 2))

        cv2.imshow('frame', frame)

        if key == ord('q'):   # quit
            print('exit')
            break


if __name__ == "__main__":
    main()
