# densepose rt with SVR correction
import math
import csv
import numpy as np
from bisect import bisect_left
from cv2 import aruco
from cv2 import cv2

import utilities as uti


# lookup table for piecewise linearization
def lookup(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    i = bisect_left(xs, x)
    k = (x - xs[i-1])/(xs[i] - xs[i-1])
    y = k*(ys[i]-ys[i-1]) + ys[i-1]

    return y


def readData():
    mk1_u_dir = 'Groundtruth/SVM_1_u.csv'
    mk1_v_dir = 'Groundtruth/SVM_1_u.csv'
    mk2_u_dir = 'Groundtruth/SVM_1_u.csv'
    mk2_v_dir = 'Groundtruth/SVM_1_u.csv'
    mk5_u_dir = 'Groundtruth/SVM_1_u.csv'
    mk5_v_dir = 'Groundtruth/SVM_1_u.csv'
    mk6_u_dir = 'Groundtruth/SVM_1_u.csv'
    mk6_v_dir = 'Groundtruth/SVM_1_u.csv'

    mk1_u = list()
    mk1_v = list()
    mk2_u = list()
    mk2_v = list()
    mk5_u = list()
    mk5_v = list()
    mk6_u = list()
    mk6_v = list()

    with open(mk1_u_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk1_u.append(row)
    with open(mk1_v_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk1_v.append(row)

    with open(mk2_u_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_u.append(row)
    with open(mk2_v_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_v.append(row)

    with open(mk5_u_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_u.append(row)
    with open(mk5_v_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_v.append(row)

    with open(mk6_u_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_u.append(row)
    with open(mk6_v_dir) as data_loader:
        reader = csv.reader(data_loader)
        for row in reader:
            mk2_v.append(row)

    mk1_u = np.reshape(mk1_u, (len(mk1_u), 2))
    mk1_v = np.reshape(mk1_v, (len(mk1_u), 2))
    mk2_u = np.reshape(mk2_u, (len(mk1_u), 2))
    mk2_v = np.reshape(mk2_v, (len(mk1_u), 2))
    mk5_u = np.reshape(mk5_u, (len(mk1_u), 2))
    mk5_v = np.reshape(mk5_v, (len(mk1_u), 2))
    mk6_u = np.reshape(mk6_u, (len(mk1_u), 2))
    mk6_v = np.reshape(mk6_v, (len(mk1_u), 2))


def main():
    readData()
    part_id = 2
    save_path = '/home/xihan/Myworkspace/lung_ultrasound/image_buffer/incoming.png'
    load_path = '/home/xihan/Myworkspace/lung_ultrasound/infer_out/incoming_IUV.png'
    tip_x = -1  # tip tracking is not necessary
    tip_y = -1
    # initial UV values
    target_u = [60, 100, 60, 100]
    target_v = [152, 167, 85, 82]

    cap = uti.initVideoStream()

    while(True):
        frame = uti.getVideoStream(cap)
        key = cv2.waitKey(33)
        cv2.imwrite(save_path, frame)
        frame, angleX, angleY = uti.detectFiducial(frame)
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
            inferred = cv2.imread(load_path)
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
