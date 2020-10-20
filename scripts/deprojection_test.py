'''
experimenting with realsense sdk python wrapper
'''
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyrealsense2 import pyrealsense2 as rs


def getNormalVector(p1, p2, p3):
    p1p2 = np.subtract(p1, p2)
    p1p3 = np.subtract(p1, p3)
    # print("p1p2: ", p1p2, "p1p3: ", p1p3)
    direction = np.cross(p1p2, p1p3)
    magnitude = np.linalg.norm(direction)
    # print(direction)
    return 0.3*direction/magnitude


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

fig = plt.figure()
plt.ion()
ax = plt.axes(projection='3d')

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

        # define triangular region of interest
        # ROI_center = [243, 177]
        center = [320, 240]
        side = 10
        # vertices of triangle P1P2P3
        # col_vec = [center[0], center[0]-side/2, center[0]+side/2]
        # row_vec = [center[1]+1.73/3*side, center[1] -
        #            1.73/6*side, center[1]-1.73/6*side]
        # vertices of the square P1P2P3P4
        col_vec = [center[0]-side/2, center[0]-side /
                   2, center[0]+side/2, center[0]+side/2]
        row_vec = [center[1]+side/2, center[1]-side /
                   2, center[1]+side/2, center[1]-side/2]

        # get corresponding xyz from uv
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_in_met = depth_frame.as_depth_frame(
        ).get_distance(center[0], center[1])
        center_xyz = rs.rs2_deproject_pixel_to_point(
            depth_intrin, center, depth_in_met)
        point_x = [round(center_xyz[0], 4)]
        point_y = [round(center_xyz[1], 4)]
        point_z = [round(center_xyz[2], 4)]
        for pnt in range(len(row_vec)):
            curr_col = round(col_vec[pnt])
            curr_row = round(row_vec[pnt])
            color_image = cv2.circle(
                color_image, (curr_col, curr_row), 2, (20, 20, 80), -1)
            depth_pixel = [curr_col, curr_row]
            depth_in_met = depth_frame.as_depth_frame().get_distance(curr_col, curr_row)
            # deprojection
            point_x.append(round(rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[0], 4))
            point_y.append(round(rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[1], 4))
            point_z.append(round(rs.rs2_deproject_pixel_to_point(
                depth_intrin, depth_pixel, depth_in_met)[2], 4))

        # find normal vector of the plane P1P2P3
        P1 = [point_x[1], point_y[1], point_z[1]]
        P2 = [point_x[2], point_y[2], point_z[2]]
        P3 = [point_x[3], point_y[3], point_z[3]]
        norm_vec = getNormalVector(P1, P2, P3)
        # print(norm_vec)
        point_x.append(round(norm_vec[0], 4))
        point_y.append(round(norm_vec[1], 4))
        point_z.append(round(norm_vec[2], 4))

        # plot points
        ax.cla()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(0.0, 1.0)
        ax.scatter3D(point_x, point_y, point_z, c=point_z, cmap='winter')
        ax.plot3D([0, norm_vec[0]], [0, norm_vec[1]], [0, norm_vec[2]], 'gray')
        plt.draw()
        plt.pause(.001)

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
