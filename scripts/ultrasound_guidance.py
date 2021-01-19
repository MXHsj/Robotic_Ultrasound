import numpy as np
from cv2 import aruco
from cv2 import cv2
import math
import time

import utilities as uti


def detectMarker(frame):
    # marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    return marker_frame, corners, ids


def main():
    part_id = 2     # 1 -> back; 2 -> front

    if part_id == 2:
        target_u = [75, 120, 75, 120]
        target_v = [152, 167, 85, 82]
    elif part_id == 1:
        target_u = [80, 80]
        target_v = [152, 85]

    cap = uti.initVideoStream(0)
    save_path = '/home/xihan/Myworkspace/lung_ultrasound/image_buffer/incoming.png'
    load_path = '/home/xihan/Myworkspace/lung_ultrasound/infer_out/incoming_IUV.png'

    while(True):
        start = time.time()
        frame = uti.getVideoStream(cap)
        cv2.imwrite(save_path, frame)
        marker_frame, corners, ids = detectMarker(frame)

        if ids is not None:     # if any marker is detected
            tip_x, tip_y, frame = uti.tipPose(corners, ids, marker_frame)
        else:
            tip_x = -1
            tip_y = -1

        try:
            inferred = cv2.imread(load_path)
        except Exception as e:
            print('image read error: '+str(e))

        if inferred is not None:
            IUV_chest = uti.getBodyPart(inferred, part_id)
            frame = uti.divide2region(frame, inferred, IUV_chest,
                                      target_u, target_v, (tip_x, tip_y))

        # frame = cv2.flip(frame, -1)     # flip back
        showFrame = cv2.resize(frame, (960, 960))
        cv2.imshow('overlay', showFrame)
        end = time.time()
        print("created mask in %.4f sec" % (end - start))

        if cv2.waitKey(1) & 0xFF == ord('q'):   # quit
            print('exiting ...')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
