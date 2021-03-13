#! /usr/bin/env python3
import numpy as np
from cv2 import aruco
from pyrealsense2 import pyrealsense2 as rs
import csv
from cv2 import cv2


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


def divide2region(frame, IUV_chest, target_u, target_v, pos):

    Radius = 15    # radius to mark circular region
    shape_color = (200, 200, 200)
    text_color = (1, 100, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    error_rec = list()

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

        if len(rcand) > 0 and len(ccand) > 0:
            cen_col = int(np.mean(ccand))  # averaging col indicies
            cen_row = int(np.mean(rcand))  # averaging row indicies
            coord = (cen_col, cen_row)
            frame = cv2.circle(frame, coord, Radius, shape_color, -1)
            cv2.putText(frame, str(reg), coord, font,
                        0.5, text_color, thickness=2)
            if pos[reg-1, 0] != -1:
                dxsq = (cen_row - pos[reg-1, 1])*(cen_row - pos[reg-1, 1])
                dysq = (cen_col - pos[reg-1, 0])*(cen_col - pos[reg-1, 0])
                error = np.sqrt(dxsq+dysq)
            else:
                error = -1
        else:
            error = -1

        error_rec.append(error)
        print("region{} error: ".format(reg)+str(error))
    return frame, error_rec


def detectMarker(frame):
    # marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    return marker_frame, corners, ids


def trackMarker(corners, ids):
    num_markers = 8
    pos = np.zeros((num_markers, 2))
    for i in range(num_markers):
        try:
            marker = corners[np.where(ids == i)[0][0]][0]
            pos[i, :] = [marker[:, 0].mean(), marker[:, 1].mean()]
        except:
            pos[i, :] = [-1, -1]      # if marker is not detected
        # print("id{} center:".format(i), pos[i-1, 0], pos[i-1, 1])
    return pos


def main():
    part_id = 2     # 1 -> posterior; 2 -> anterior
    if part_id == 2:
        target_u = [60, 100, 60, 100]
        target_v = [162, 162, 95, 95]
        # target_u = [60, 100, 60, 100, 60, 100, 60, 100]
        # target_v = [142, 157, 180, 180, 95, 92, 60, 65]
        # target_u = [60, 100, 60, 100]
        # target_v = [155, 155, 105, 105]
    elif part_id == 1:
        target_u = [80, 80]
        target_v = [167, 82]

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    save_path = '/home/xihan/Myworkspace/lung_ultrasound/image_buffer/incoming.png'
    load_path = '/home/xihan/Myworkspace/lung_ultrasound/infer_out/incoming_IUV.png'
    data_record = '/home/xihan/Myworkspace/lung_ultrasound/scripts/overlay_error.csv'
    file_out = open(data_record, 'w')
    writer = csv.writer(file_out)

    sample_size = 100
    sample_count = 0
    isRecording = False
    while(sample_count < sample_size):
        frames = pipeline.wait_for_frames()

        # align depth to color frame
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = color_image[:, 80:560]    # crop to square patch
        frame = color_image
        cv2.imwrite(save_path, frame)

        frame, corners, ids = detectMarker(frame)
        pos = trackMarker(corners, ids)

        try:
            inferred = cv2.imread(load_path)
        except Exception as e:
            print('error: '+str(e))

        if inferred is not None:
            IUV_chest = getBodyPart(inferred, part_id)
            frame, errors = divide2region(
                frame, IUV_chest, target_u, target_v, pos)

            if errors[0] != -1 and errors[1] != -1 and errors[2] != -1 and errors[3] != -1:
                if isRecording:
                    row2write = [errors[0], errors[1], errors[2], errors[3]]
                    writer.writerow(row2write)
                    sample_count += 1
        else:
            pass

        # showFrame = cv2.resize(frame, (720, 720))
        cv2.imshow('overlay', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):   # quit
            print('exiting ...')
            break
        elif key == ord('s'):
            print('start recording')
            isRecording = True
        elif key == ord('e'):
            print('end recording')
            isRecording = False

    cv2.destroyAllWindows()
    file_out.close()


if __name__ == "__main__":
    main()
