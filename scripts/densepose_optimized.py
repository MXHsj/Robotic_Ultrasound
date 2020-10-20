# densepose rt with SVR correction
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from cv2 import aruco
from joblib import dump, load
from cv2 import cv2

import utilities as uti


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

    cap = uti.initVideoStream()

    while(True):
        frame = uti.getVideoStream(cap)
        key = cv2.waitKey(33)
        cv2.imwrite(save_path, frame)
        frame, angleX = uti.detectFiducial(frame)   # detect fiducial marker
        frame, pos = uti.detectMarker(frame)        # detect overlay marker

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
            IUV_chest = uti.getBodyPart(inferred, part_id)
            frame = uti.divide2region(frame, inferred, IUV_chest,
                                      target_u, target_v, (tip_x, tip_y))
            UV = uti.getUV(IUV_chest, pos)
        else:
            UV = -1*np.ones((4, 2))

        cv2.imshow('frame', frame)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
