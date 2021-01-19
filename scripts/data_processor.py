import os
import csv
from cv2 import cv2
import utilities as uti

img_dir = '/home/xihan/Myworkspace/lung_ultrasound/video2images'
infer_dir = '/home/xihan/Myworkspace/lung_ultrasound/infer_out'
u_record = '/home/xihan/Myworkspace/lung_ultrasound/scripts/angle_u.csv'
v_record = '/home/xihan/Myworkspace/lung_ultrasound/scripts/angle_v.csv'

list = os.listdir(img_dir)

n_sample = len(list)
n_marker = 8
part_id = 2     # detect anterior

# cv2.startWindowThread()
# cv2.namedWindow("preview")

for it in range(n_sample-10):
    # it = 150
    img_name = "image"+str(it+1)+".jpg"
    mask_name = "image"+str(it+1)+"_IUV.png"

    frame = cv2.imread(os.path.join(img_dir, img_name))
    frame, angleX, angleY, angleZ = uti.detectFiducial(frame)
    frame, pos = uti.detectMarker(
        frame, n_marker)        # detect overlay marker
    # print(angleX)
    # print(pos)

    mask = cv2.imread(os.path.join(infer_dir, mask_name))
    IUV_chest = uti.getBodyPart(mask, part_id)
    UV = uti.getUV(IUV_chest, pos)
    UV[UV == 0] = -1
    # print(UV)

    # save v
    with open(u_record, 'a') as file_out:
        writer = csv.writer(file_out)
        for i in range(len(UV)):
            writer.writerow(
                [i+1, angleX, angleY, angleZ, UV[i, 0]])
    # save u
    with open(v_record, 'a') as file_out:
        writer = csv.writer(file_out)
        for i in range(len(UV)):
            writer.writerow(
                [i+1, angleX, angleY, angleZ, UV[i, 1]])

    # cv2.imshow('preview', frame)
    # cv2.imshow('preview', mask)
    # key = cv2.waitKey(0)

    # cv2.destroyAllWindows()


print('finish')
