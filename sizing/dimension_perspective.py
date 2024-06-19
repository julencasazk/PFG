import pyrealsense2 as rs
import time
import numpy as np
import cv2
import open3d as o3d

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

spatial = rs.temporal_filter()
reference_frame = None
try:
    while not cv2.waitKey(1) & 0xFF == ord('q'):

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_profile = depth_frame.get_profile()
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        # depth_frame = spatial.process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #depth_image[depth_image > 1000] = 0

        comparison_frame = depth_image.copy()

        if (cv2.waitKey(1) & 0xFF) == ord(' '):
            reference_frame = depth_image.copy()


        if reference_frame is not None:

            reference_o3d = o3d.geometry.Image(reference_frame.astype(np.uint16))
            comparison_o3d = o3d.geometry.Image(comparison_frame.astype(np.uint16))

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

            # Compute the difference between the two point clouds
            threshold = 0.01
            distances = comparison_pcd.compute_point_cloud_distance(reference_pcd)
            mask = np.asarray(distances) > threshold
            comparison_pcd.points = o3d.utility.Vector3dVector(np.asarray(comparison_pcd.points)[mask])
            cl, ind = comparison_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.5)


            
            # Visualize the point cloud
            if (cv2.waitKey(1) & 0xFF) == ord('f'):
                if len(cl.points) > 0:
                    before = time.time()
                    labels = np.asarray(cl.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
                    after = time.time()
                    print(f"Clustering time: ", after - before)

                    unique, counts = np.unique(labels, return_counts=True)
                    max_cluster_index = np.argmax(counts)
                    # Biggest cluster, it should be the box after filtering
                    max_cluster_label = unique[max_cluster_index]
                    max_cluster = cl.select_by_index(np.where(labels == max_cluster_label)[0])
                    bounding_box = max_cluster.get_minimal_oriented_bounding_box(robust=True)
                    #bounding_box, _ = max_cluster.compute_convex_hull()
                    plane_models = []
                    planes = []
                    rectangles = []
                    current_pcl = max_cluster
                    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    for i in range(3):
                        if len(current_pcl.points) > 0:
                            plane_model, inliers = current_pcl.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                            inlier_cloud = current_pcl.select_by_index(inliers)
                            rectangle = inlier_cloud.get_minimal_oriented_bounding_box()
                            rectangle.color = colors[i]
                            rectangles.append(rectangle)
                            inlier_cloud.paint_uniform_color(colors[i])
                            outliers_cloud = current_pcl.select_by_index(inliers, invert=True)
                            planes.append(inlier_cloud)
                            plane_models.append(plane_model)
                            current_pcl = outliers_cloud

                    #hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(bou33232nding_box)
                    #hull_ls.paint_uniform_color([1, 0, 0])
                    bounding_box.color = (1, 0, 0)
                    # Compute normals for the point cloud
                    """ max_cluster.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)) """

                    # Detect planar patches
                    """ oboxes = max_cluster.detect_planar_patches(
                        normal_variance_threshold_deg=5,
                        coplanarity_deg=75,
                        outlier_ratio=0.75,
                        min_plane_edge_length=0,
                        min_num_points=0,
                        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
                    
                    print("Detected {} patches".format(len(oboxes))) """

                    # Get the three largest planes
                    """ main_planes = sorted(oboxes, key=lambda obox: obox.extent[2], reverse=True)[:3] """

                    """  points_in_planes = []
                    for obox in main_planes:
                        print(f"Plane: {obox}") """

                    """ geometries = []
                    for obox in main_planes:
                        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
                        mesh.paint_uniform_color(obox.color)
                        geometries.append(mesh)
                        geometries.append(obox)
                    geometries.append(bounding_box)
                    geometries.append(max_cluster) """
                    #o3d.visualization.draw_geometries(geometries)
                    planes.append(bounding_box)
                    planes.extend(rectangles)
                    # Planes: The points that belong to the three or two planes
                    # comparison_pcd: All the points after filtering
                    
                    o3d.visualization.draw_geometries(planes)
                    o3d.visualization.draw_geometries([comparison_pcd, bounding_box])

                    dimensions = np.array(bounding_box.extent)

                    sorted_dimensions = np.sort(dimensions)
                    length, width, height = sorted_dimensions[::-1]  # Ordena de mayor a menor

                    print("Length:", length)
                    print("Width:", width)
                    print("Height:", height)
                    print("Volume:", length * width * height)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(comparison_frame, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if reference_frame is not None:
            ref_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(reference_frame, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((ref_depth_colormap, depth_colormap))
        else:
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

        #def mouse_event_callback(event, x, y, flags, param):
        #   if event == cv2.EVENT_MOUSEMOVE:
        #        depth = comparison_frame[y, x]
        #        print(f"Depth at pixel ({x}, {y}): {depth}")

        #print(f"Depth at pixel (320, 240): {depth_image[240, 320]}")

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.setMouseCallback('RealSense', mouse_event_callback)
        cv2.imshow('RealSense', images)
        

finally:

    # Stop streaming
    pipeline.stop()
