'''
experimenting with realsense sdk python wrapper
'''
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
from pyrealsense2 import pyrealsense2 as rs


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
# dec_filter = rs.decimation_filter()     # reduce depth frame density
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

        # filtered = dec_filter.process(depth_frame)
        filtered = spat_filter.process(depth_frame)
        filtered = temp_filter.process(filtered)

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(filtered.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # draw overlay on color frame
        ROI_center = [320, 240]
        increment = 1
        start_point = (ROI_center[0]-increment, ROI_center[1]-increment)
        end_point = (ROI_center[0]+increment, ROI_center[1]+increment)
        color_image = cv2.rectangle(
            color_image, start_point, end_point, (20, 20, 80), 2)

        # find surface normal using depth gradient
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_center = depth_frame.as_depth_frame(
        ).get_distance(ROI_center[0], ROI_center[1])

        depth_left = depth_frame.as_depth_frame(
        ).get_distance(ROI_center[0]-increment, ROI_center[1])

        depth_right = depth_frame.as_depth_frame(
        ).get_distance(ROI_center[0]+increment, ROI_center[1])

        depth_up = depth_frame.as_depth_frame(
        ).get_distance(ROI_center[0], ROI_center[1]-increment)

        depth_down = depth_frame.as_depth_frame(
        ).get_distance(ROI_center[0], ROI_center[1]+increment)

        dzdx = (depth_right - depth_left)/(2.0)
        dzdy = (depth_down - depth_up)/(2.0)

        direction = [-dzdx, -dzdy, 1.0]
        magnitude = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
        norm_vec = direction/magnitude

        ROI_center_xyz = rs.rs2_deproject_pixel_to_point(
            depth_intrin, ROI_center, depth_center)
        point_x = [round(ROI_center_xyz[0], 4), norm_vec[0]]
        point_y = [round(ROI_center_xyz[1], 4), norm_vec[1]]
        point_z = [round(ROI_center_xyz[2], 4), norm_vec[2]]

        # plot points in 3D
        ax.cla()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_xlim(-0.06, 0.06)
        ax.set_ylim(-0.06, 0.06)
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
