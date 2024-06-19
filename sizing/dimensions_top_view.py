import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from opcua import Client

def main(ip="localhost"):
    # Conectar al servidor OPC UA
    client = Client(f"opc.tcp://{ip}:4840")
    try:
        client.connect()
        print("Connected to OPC UA server")
    except Exception as e:
        print(f"Failed to connect to OPC UA server: {e}")
        return

    try:
        # Obtener nodos del servidor
        activate_cam = client.get_node("ns=2;i=25")
        if not activate_cam:
            print("Node activate_cam not found")
            return

        box_dimensions = client.get_node("ns=2;i=20")
        if not box_dimensions:
            print("Node box_dimensions not found")
            return

        # Configurar los flujos de profundidad y color
        pipeline = rs.pipeline()
        config = rs.config()

        # Obtener la línea de productos del dispositivo para configurar una resolución compatible
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

        # Iniciar la transmisión
        pipeline.start(config)

        spatial = rs.temporal_filter()
        reference_frame = None

        lengths = []
        widths = []
        heights = []
        areas = []
        max_index = 0
        reference_taken = False
        try:
            while not cv2.waitKey(1) & 0xFF == ord('q'):
                activate_cam_value = activate_cam.get_value()
                # Wait for coherent frames
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_profile = depth_frame.get_profile()
                intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

                # Convertir imágenes a arrays de numpy
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_image[depth_image > 1000] = 0

                comparison_frame = depth_image.copy()

                if activate_cam_value and not reference_taken:
                    reference_frame = depth_image.copy()
                    reference_taken = True

                elif not activate_cam_value and reference_taken:
                    if len(lengths) > 0:
                        max_index = areas.index(max(areas))
                        print(f"Contents from all the boxes:")
                        print(f"Volumes: {areas}")
                        print(f"Length: {lengths}")
                        print(f"Width: {widths}")
                        print(f"Width: {widths[max_index]}")
                        print(f"Length: {lengths[max_index]}")
                        print(f"Height: {heights[max_index]}")
                        print(f"Volume: {lengths[max_index] * widths[max_index] * heights[max_index]}")
                        box_dimensions.set_value([widths[max_index], lengths[max_index], heights[max_index]])
                        lengths = []
                        widths = []
                        heights = []
                        areas = []

                    reference_taken = False

                if reference_frame is not None and activate_cam_value:
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

                    reference_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        reference_o3d,
                        o3d_intrinsics
                    )

                    comparison_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        comparison_o3d,
                        o3d_intrinsics
                    )

                    floor_plane_model, inliers = comparison_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                    floor_plane = comparison_pcd.select_by_index(inliers)

                    threshold = 0.05
                    distances = comparison_pcd.compute_point_cloud_distance(reference_pcd)
                    mask = np.asarray(distances) > threshold
                    comparison_pcd.points = o3d.utility.Vector3dVector(np.asarray(comparison_pcd.points)[mask])
                    cl, ind = comparison_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.5)

                    if len(cl.points) > 0:
                        if len(cl.points) > 500:
                            plane_model, inliers = cl.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                            plane_points = cl.select_by_index(inliers)
                            bounding_box = plane_points.get_minimal_oriented_bounding_box(robust=True)

                            height = abs(plane_model[3] - floor_plane_model[3]) / np.sqrt(plane_model[0]**2 + plane_model[1]**2 + plane_model[2]**2)

                            bounding_box.color = (1, 0, 0)

                            dimensions = np.array(bounding_box.extent)

                            length = max(dimensions)
                            for i in range(3):
                                if dimensions[i] == length:
                                    dimensions[i] = 0
                                    break
                            width = max(dimensions)
                            print(f"dimensions: {dimensions}")
                            lengths.append(length)
                            widths.append(width)
                            heights.append(height)
                            areas.append(length * width)

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

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

        finally:
            # Detener la transmisión
            pipeline.stop()
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Desconectar el cliente OPC UA
        client.disconnect()
        print("Disconnected from OPC UA server")

if __name__ == "__main__":
    main("10.172.7.140")
