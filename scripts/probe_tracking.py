import numpy as np
from cv2 import aruco
from cv2 import cv2
import math
import time

camera_matrix = np.array(
    [[662.1790, 0.0, 322.3619], [0.0, 662.8344, 252.0131], [0.0, 0.0, 1.0]])
dist_coeff = np.array([0.0430651, -0.1456001, 0.0, 0.0])


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)

    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


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
    Radius = 20    # radius to mark circular region
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


def detectMarker(frame):
    # marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    return marker_frame, corners, ids


def tipPose(corners, ids, frame):
    probeMarkerID = 3
    Pc = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  # world -> camera
    Ptip_left = [[-0.06], [0.03], [-0.1], [1.0]]
    Ptip_right = [[-0.06], [-0.03], [-0.1], [1.0]]
    Ptip_center = [[-0.06], [0.0], [-0.1], [1.0]]   # x, y, z   0.0, 0.0, -0.24
    try:
        probe_marker = corners[np.where(ids == probeMarkerID)[0][0]]
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
            probe_marker, 0.048, camera_matrix, dist_coeff)
        # axis_frame = aruco.drawAxis(
        #     frame, camera_matrix, dist_coeff, rvecs, tvecs, 0.05)
        rmat = cv2.Rodrigues(rvecs)[0]  # convert rot vector to rot matrix
        tvecs = np.transpose(tvecs)
        # marker uv -> marker xyz
        Tcam2marker = [[rmat[0, 0], rmat[0, 1], rmat[0, 2], tvecs[0, 0, 0]],
                       [rmat[1, 0], rmat[1, 1], rmat[1, 2], tvecs[1, 0, 0]],
                       [rmat[2, 0], rmat[2, 1], rmat[2, 2], tvecs[2, 0, 0]],
                       [0, 0, 0, 1]]
        xyz2uv = np.matmul(camera_matrix, Pc)
        # marker xyz -> tip xyz
        tip_left_pose = np.matmul(Tcam2marker, Ptip_left)
        tip_right_pose = np.matmul(Tcam2marker, Ptip_right)
        tip_center_pose = np.matmul(Tcam2marker, Ptip_center)
        # tip xyz -> tip uv
        tip_left_pixel = np.matmul(xyz2uv, tip_left_pose)
        tip_left_pixel = tip_left_pixel/tip_left_pixel[2, 0]

        tip_right_pixel = np.matmul(xyz2uv, tip_right_pose)
        tip_right_pixel = tip_right_pixel/tip_right_pixel[2, 0]

        tip_center_pixel = np.matmul(xyz2uv, tip_center_pose)
        tip_center_pixel = tip_center_pixel/tip_center_pixel[2, 0]

        x_curr = int(tip_center_pixel[0])
        y_curr = int(tip_center_pixel[1])

        x_right = int(tip_right_pixel[0])
        y_right = int(tip_right_pixel[1])

        x_left = int(tip_left_pixel[0])
        y_left = int(tip_left_pixel[1])

        frame = cv2.line(frame, (x_left, y_left),
                         (x_right, y_right), (1, 100, 1), thickness=2)
        frame = cv2.circle(frame, (x_curr, y_curr), 3, (255, 1, 1), -1)

    except Exception as e:
        print('probe detection error: '+str(e))
        x_curr = -1
        y_curr = -1

    return x_curr, y_curr, frame


def initVideoStream():
    cap = cv2.VideoCapture(2)
    focus = 0               # min: 0, max: 255, increment:5
    cap.set(28, focus)      # manually set focus
    return cap


def getVideoStream(cap):
    # patch_size = 480
    _, frame = cap.read()
    # frame = cv2.resize(frame, (patch_size, patch_size))
    frame = frame[:, 80:560]
    # frame = cv2.flip(frame, -1)     # flip the frame
    return frame


def main():
    part_id = 2     # 1 -> back; 2 -> front

    if part_id == 2:
        target_u = [75, 120, 75, 120]
        target_v = [152, 167, 85, 82]
    elif part_id == 1:
        target_u = [80, 80]
        target_v = [152, 85]

    cap = initVideoStream()
    save_path = '/home/xihan/Myworkspace/lung_ultrasound/image_buffer/incoming.png'

    while(True):
        start = time.time()
        frame = getVideoStream(cap)
        cv2.imwrite(save_path, frame)
        marker_frame, corners, ids = detectMarker(frame)

        if ids is not None:     # if any marker is detected
            tip_x, tip_y, frame = tipPose(corners, ids, marker_frame)
        else:
            tip_x = -1
            tip_y = -1

        try:
            inferred = cv2.imread(
                '/home/xihan/Myworkspace/lung_ultrasound/infer_out/incoming_IUV.png')
        except Exception as e:
            print('image read error: '+str(e))

        if inferred is not None:
            IUV_chest = getBodyPart(inferred, part_id)
            frame = divide2region(frame, inferred, IUV_chest,
                                  target_u, target_v, (tip_x, tip_y))

        # frame = cv2.flip(frame, -1)     # flip back
        cv2.imshow('overlay', frame)
        end = time.time()
        print("created mask in %.4f sec" % (end - start))

        if cv2.waitKey(1) & 0xFF == ord('q'):   # quit
            print('exiting ...')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
