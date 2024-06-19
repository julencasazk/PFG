import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

def main():

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Color sensor not detected, exiting...")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    reference_frame = None
    comparison_pcd = None
    filtered_pcd = None

    print("Press 'Space' to capture reference frame. Press 'F' to capture frame with box and calculate dimensions. Press 'J' to visualize point clouds. Press 'Q' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_image[depth_image > 1000] = 0

            comparison_frame = depth_image.copy()

            key = cv2.waitKey(1)
            
            if key == ord(' '):  # Capture reference frame
                reference_frame = depth_image.copy()
                print("Reference frame captured.")

            if reference_frame is not None:
                reference_o3d = o3d.geometry.Image(reference_frame.astype(np.uint16))
                comparison_o3d = o3d.geometry.Image(comparison_frame.astype(np.uint16))

                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    intrinsics.width,
                    intrinsics.height,
                    intrinsics.fx,
                    intrinsics.fy,
                    intrinsics.ppx,
                    intrinsics.ppy
                )

                # Create the point cloud
                reference_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    reference_o3d,
                    o3d_intrinsics
                )

                comparison_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    comparison_o3d,
                    o3d_intrinsics
                )

                # Detect the largest plane in the point cloud
                floor_plane_model, inliers = comparison_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                floor_plane = comparison_pcd.select_by_index(inliers)
                
                # Filter points that are not in the reference plane
                threshold = 0.06
                distances = comparison_pcd.compute_point_cloud_distance(reference_pcd)
                mask = np.asarray(distances) > threshold
                filtered_points = np.asarray(comparison_pcd.points)[mask]
                filtered_pcd = o3d.geometry.PointCloud()
                filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                filtered_pcd, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.5)

                if key == ord('f') and len(filtered_pcd.points) > 0:
                    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=50)
                    plane_points = filtered_pcd.select_by_index(inliers)
                    bounding_box = plane_points.get_oriented_bounding_box()

                    # Calculate the height of the box as the distance between the parallel planes
                    height = abs(plane_model[3] - floor_plane_model[3]) / np.sqrt(plane_model[0]**2 + plane_model[1]**2 + plane_model[2]**2)

                    # Calculate the dimensions of the box
                    dimensions = list(bounding_box.extent)  # Convert to list to make it mutable
                    length = max(dimensions)
                    for i in range(3):
                        if dimensions[i] == length:
                            dimensions[i] = 0
                            break
                    width = max(dimensions)

                    print(f"Box dimensions (L x W x H): {length:.6f} x {width:.6f} x {height:.6f} meters")

            if key == ord('j') and comparison_pcd is not None and filtered_pcd is not None:
                o3d.visualization.draw_geometries([comparison_pcd])
                o3d.visualization.draw_geometries([filtered_pcd])

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(comparison_frame, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            if reference_frame is not None:
                ref_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(reference_frame, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((ref_depth_colormap, depth_colormap))
            else:
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            if key == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
