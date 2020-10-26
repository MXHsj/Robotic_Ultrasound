'''
experimenting with realsense sdk python wrapper
'''
import csv
import numpy as np
from cv2 import cv2
from pyrealsense2 import pyrealsense2 as rs


def getNormalVector(p0, p1, p2):
    p1p0 = np.subtract(p1, p0)
    p2p0 = np.subtract(p2, p0)
    direction = np.cross(p1p0, p2p0)
    if direction[2] < 0:
        direction[0] = - direction[0]
        direction[1] = - direction[1]
        direction[2] = - direction[2]
    # magnitude = np.linalg.norm(direction)
    return direction


def getSurfaceNormal(point_x, point_y, point_z):
    # find normal vector of the plane P1P2P3P4
    P0 = [point_x[0], point_y[0], point_z[0]]
    P1 = [point_x[1], point_y[1], point_z[1]]
    P2 = [point_x[2], point_y[2], point_z[2]]
    P3 = [point_x[3], point_y[3], point_z[3]]
    P4 = [point_x[4], point_y[4], point_z[4]]
    P5 = [point_x[5], point_y[5], point_z[5]]
    P6 = [point_x[6], point_y[6], point_z[6]]
    P7 = [point_x[7], point_y[7], point_z[7]]
    P8 = [point_x[8], point_y[8], point_z[8]]
    # print(" P0: \n", P0, "\n P1: \n", P1, "\n P2: \n",
    #       P2, "\n P3: \n", P3, "\n P4: \n", P4)
    norm1 = getNormalVector(P0, P1, P2)
    norm2 = getNormalVector(P0, P2, P3)
    norm3 = getNormalVector(P0, P3, P4)
    norm4 = getNormalVector(P0, P4, P5)
    norm5 = getNormalVector(P0, P5, P6)
    norm6 = getNormalVector(P0, P6, P7)
    norm7 = getNormalVector(P0, P7, P8)
    norm8 = getNormalVector(P0, P1, P8)

    norm_vec = norm1+norm2+norm3+norm4+norm5+norm6+norm7+norm8
    norm_vec = 0.005*norm_vec/np.linalg.norm(norm_vec)
    # print(norm_vec)
    return norm_vec


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def ROIshape():
    center = [320, 240]  # center pixel
    side = 5            # side length of the square
    # square region
    col_vec = [center[0],
               center[0]-side/2,
               center[0]-side/2,
               center[0]-side/2,
               center[0],
               center[0]+side/2,
               center[0]+side/2,
               center[0]+side/2,
               center[0]]

    row_vec = [center[1],
               center[1]+side/2,
               center[1],
               center[1]-side/2,
               center[1]-side/2,
               center[1]-side/2,
               center[1],
               center[1]+side/2,
               center[1]+side/2]
    return col_vec, row_vec


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

# depth filter
dec_filter = rs.decimation_filter()     # reduce depth frame density
temp_filter = rs.temporal_filter()      # edge-preserving spatial smoothing
spat_filter = rs.spatial_filter()       # reduce temporal noise

col_vec, row_vec = ROIshape()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # align depth to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        filtered = dec_filter.process(depth_frame)
        filtered = spat_filter.process(filtered)
        filtered = temp_filter.process(filtered)

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(filtered.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # get corresponding xyz from uv
        point_x = []
        point_y = []
        point_z = []
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        for pnt in range(len(row_vec)):
            curr_col = round(col_vec[pnt])
            curr_row = round(row_vec[pnt])
            color_image = cv2.circle(
                color_image, (curr_col, curr_row), 1, (30, 90, 30), -1)
            depth_pixel = [curr_col, curr_row]
            depth_in_met = depth_frame.as_depth_frame().get_distance(curr_col, curr_row)
            # deprojection
            x = rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[0]
            y = rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[1]
            z = rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[2]
            point_x.append(x)
            point_y.append(y)
            point_z.append(z)

        norm_vec = getSurfaceNormal(point_x, point_y, point_z)
        point_x.append(my_floor(norm_vec[0], 3))
        point_y.append(my_floor(norm_vec[1], 3))
        point_z.append(my_floor(norm_vec[2], 3))

        # write data into csv file
        with open('./surface_normal.csv', 'a') as file_out:
            writer = csv.writer(file_out)
            writer.writerow(point_x)
            writer.writerow(point_y)
            writer.writerow(point_z)

        # Stack both images horizontally
        # images = np.vstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()
