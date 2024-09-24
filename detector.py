"""
Boat Detection and Tracking System using LiDAR Point Cloud Data

This Python ROS2 node detects and tracks boats using point cloud data from a LiDAR sensor. The system implements various clustering and tracking algorithms to identify, cluster, and track boats in real time. Major functionalities include:

Clustering Algorithms: Implements DBSCAN, HDBSCAN, and a custom 2D clustering approach by squashing the Z-axis for boat detection.
Kalman Filter: A Kalman filter is used to track the position and movement of detected boats, allowing prediction and correction based on new data.
Visualization: Real-time visualization of detected boats, represented in a top-down view, with altitude-based color-coding for the point cloud data. The visualization updates continuously in a matplotlib window.
Detection Methods:
DBSCAN: Detects clusters in the point cloud and groups points into potential boats based on distance and density.
HDBSCAN: An alternative clustering algorithm that adapts based on the density of points.
2D Clustering: Projects the point cloud onto the XY-plane, ignoring height (Z) for more stable clustering.
Line Detection (RANSAC): Detects lines in point cloud data that could represent boats, especially useful for long, narrow vessels.
Key methods include:

create_kalman_filter(initial_position): Initializes a Kalman filter for tracking detected boats.
listener_callback(msg): Processes incoming LiDAR point cloud data and runs the boat detection and tracking pipeline.
detect_boats(points): Selects and runs the appropriate detection method based on the configured clustering approach.
update_tracks(detected_boats): Updates or creates new boat tracks based on detected boats using the Hungarian algorithm for assignment.
visualize(points): Continuously visualizes point cloud data and tracked boats in a 2D top-down view, with color-coding based on altitude (Z-coordinate).
This node is designed for real-time boat tracking in environments where a LiDAR sensor is installed on a boat, providing a detailed view of nearby vessels.

Necessary installations:
ROS2 and ROS2 Packages: https://docs.ros.org/en/humble/Installation.html
Python Libraries:
NumPy: pip install numpy
Matplotlib: pip install matplotlib
FilterPy (for kalman filter): pip install filterpy
Scikit-learn (For DBSCAN clustering): pip install scikit-learn
HDBSCAN (For the HDBSCAN clustering): pip install hdbscan
SciPy: (For the Hungarian algorithm): pip install scipy
CV2: pip install opencv-python
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Add this import
import time
from filterpy.kalman import KalmanFilter

# Import clustering algorithms
from sklearn.cluster import DBSCAN
import hdbscan  # Install via pip install hdbscan

# Import for line detection and PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

# Import OpenCV for image processing
import cv2  # Install via pip install opencv-python

import numpy as np
from scipy.spatial import cKDTree

class BoatDetector(Node):
    def __init__(self):
        super().__init__('boat_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            'lidar_pointcloud',
            self.listener_callback,
            10)
        
        # Initialize tracking variables
        self.next_boat_id = 1
        self.tracked_boats = {}  # boat_id: {'kf': KalmanFilter, 'last_seen': timestamp, 'points': np.array}
        self.tracking_threshold = 15.0  # Maximum distance to consider same boat (meters)
        self.max_disappeared_time = 5.0  # Maximum time (seconds) to retain a boat without detection

        # Create the figure and axes once
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.running = True

        # Set fixed axis limits
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Top-Down View of Detected Boats with Tracking')
        self.ax.grid(True)

        plt.ion()
        plt.show()

        # Detection method selection
        # 0: DBSCAN
        # 1: HDBSCAN
        # 2: 2D Clustering after squashing Z
        # 3: Line-based detection with gap filling
        # 4: Altitude-based Planar Clustering
        # 5: PCA-Based Shape Analysis
        self.DETECTION_METHOD = 0  # Change this value to select different methods

    def create_kalman_filter(self, initial_position):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 0.1  # Time interval between measurements
        kf.x = np.array([initial_position[0], 0, initial_position[1], 0])  # [x, vx, y, vy]

        # Constant position model (since boats may appear stationary relative to moving sensor)
        kf.F = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])
        kf.P *= 1000.0
        kf.R *= 10.0  # Increased measurement noise
        kf.Q = np.eye(4) * 0.1  # Increased process noise
        return kf

    def handle_close(self, event):
        self.running = False
        rclpy.shutdown()

    def listener_callback(self, msg):
        if not self.running:
            return
        points = self.pointcloud2_to_array(msg)
        filtered_points = self.filter_points(points)
        boats = self.detect_boats(filtered_points)
        self.update_tracks(boats)
        self.visualize(filtered_points)

    def pointcloud2_to_array(self, cloud_msg):
        cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
        points = cloud_arr.reshape(-1, 3)
        return points

    def filter_points(self, points):
        # Exclude points within the own boat rectangle
        x_min, x_max = -10.0, 2.0
        y_min, y_max = -2.0, 2.0
        mask = (
            (points[:, 0] < x_min) | (points[:, 0] > x_max) |
            (points[:, 1] < y_min) | (points[:, 1] > y_max)
        )
        points = points[mask]

        # Apply height filter to exclude points above 75 meters
        z_max_threshold = 75.0  # Height limit in meters
        z_min_threshold = 0.3   # Only consider points above water level
        points = points[(points[:, 2] > z_min_threshold) & (points[:, 2] < z_max_threshold)]
        return points

    def detect_boats(self, points):
        if points.size == 0:
            return []
        # Select detection method
        if self.DETECTION_METHOD == 0:
            boats = self.detect_boats_dbscan(points)
        elif self.DETECTION_METHOD == 1:
            boats = self.detect_boats_hdbscan(points)
        elif self.DETECTION_METHOD == 2:
            boats = self.detect_boats_2d_clustering(points)
        elif self.DETECTION_METHOD == 3:
            boats = self.detect_boats_line_detection(points)
        elif self.DETECTION_METHOD == 4:
            boats = self.detect_boats_altitude_clustering(points)
        elif self.DETECTION_METHOD == 5:
            boats = self.detect_boats_pca(points)
        else:
            self.get_logger().error(f"Invalid DETECTION_METHOD: {self.DETECTION_METHOD}")
            boats = []
        return boats

    def detect_boats_dbscan(self, points):
        # DBSCAN parameters
        eps = 5.0  # Increased eps for more lenient clustering
        min_samples = 10  # Reduced min_samples to detect smaller clusters

        # Use all three dimensions for clustering
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(points)

        boats = []
        for label in set(labels):
            if label == -1:
                continue  # Noise
            cluster_points = points[labels == label]
            
            # Basic filtering
            if self.is_potential_boat(cluster_points):
                centroid = np.mean(cluster_points[:, :2], axis=0)
                boats.append({'points': cluster_points, 'centroid': centroid})

        # Merge nearby detections
        merged_boats = self.merge_detections(boats)
        return merged_boats

    def is_potential_boat(self, cluster_points):
        # Calculate cluster dimensions
        x_min, x_max = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
        y_min, y_max = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
        
        length = x_max - x_min
        width = y_max - y_min

        # Very basic size check
        if length < 1.0 or width < 0.5:  # Minimum size to consider
            return False

        # Check number of points
        if len(cluster_points) < 5:  # Minimum number of points to consider
            return False

        return True

    def merge_detections(self, detections, distance_threshold=10.0):
        if not detections:
            return []

        merged = []
        while detections:
            base = detections.pop(0)
            base_centroid = base['centroid']
            
            i = 0
            while i < len(detections):
                if np.linalg.norm(base_centroid - detections[i]['centroid']) < distance_threshold:
                    # Merge this detection with base
                    base['points'] = np.vstack((base['points'], detections[i]['points']))
                    base_centroid = np.mean(base['points'][:, :2], axis=0)
                    base['centroid'] = base_centroid
                    detections.pop(i)
                else:
                    i += 1
            
            merged.append(base)

        return merged

    def detect_boats_hdbscan(self, points):
        # HDBSCAN parameters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
        labels = clusterer.fit_predict(points[:, :2])

        boats = self.process_clusters(points, labels)
        return boats

    def detect_boats_2d_clustering(self, points):
        # Squash Z-axis (project points onto XY plane)
        points_2d = points.copy()
        points_2d[:, 2] = 0  # Set Z to zero

        # Clustering in 2D
        clusterer = DBSCAN(eps=5.0, min_samples=10)
        labels = clusterer.fit_predict(points_2d[:, :2])

        boats = self.process_clusters(points, labels)
        return boats

    def detect_boats_line_detection(self, points):
        # Project points onto XY plane
        points_2d = points[:, :2]

        # Create a 2D grid map
        grid_resolution = 0.5  # Adjust as needed
        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

        x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)

        grid, x_edges, y_edges = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=[x_bins, y_bins])

        # Generate binary image
        binary_image = (grid > 0).astype(np.uint8)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Apply Canny edge detection
        edges = cv2.Canny(dilated_image * 255, 50, 150)

        # Find coordinates of edges
        edge_indices = np.column_stack(np.nonzero(edges))

        # Map back to world coordinates
        x_coords = x_edges[edge_indices[:, 0]]
        y_coords = y_edges[edge_indices[:, 1]]
        edge_points = np.column_stack((x_coords, y_coords))

        # Run RANSAC on edge points
        boats = []
        if len(edge_points) < 2:
            return boats

        # Parameters
        min_samples = 2
        residual_threshold = 1.0
        max_trials = 100

        # Apply RANSAC to find lines
        ransac = RANSACRegressor(residual_threshold=residual_threshold, max_trials=max_trials)
        poly = PolynomialFeatures(degree=1)
        X = edge_points[:, 0].reshape(-1, 1)
        y = edge_points[:, 1]

        try:
            ransac.fit(poly.fit_transform(X), y)
            inlier_mask = ransac.inlier_mask_
            boat_points = edge_points[inlier_mask]

            # Process the detected line as a boat if it meets criteria
            if len(boat_points) >= 10:
                # Compute centroid
                centroid = np.mean(boat_points, axis=0)
                boats.append({'points': boat_points, 'centroid': centroid})
        except Exception as e:
            self.get_logger().error(f"Line detection error: {e}")

        return boats

    def detect_boats_altitude_clustering(self, points):
        # Define altitude slicing parameters
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        z_interval = 5.0  # Adjust as needed

        z_slices = np.arange(z_min, z_max + z_interval, z_interval)
        slice_indices = np.digitize(points[:, 2], z_slices)

        clusters = []
        for i in range(1, len(z_slices)):
            # Points in the current slice
            slice_mask = slice_indices == i
            slice_points = points[slice_mask]

            if slice_points.size == 0:
                continue

            # Perform 2D clustering on the slice
            clusterer = DBSCAN(eps=2.5, min_samples=5)
            labels = clusterer.fit_predict(slice_points[:, :2])

            # Store clusters with slice information
            for label in set(labels):
                if label == -1:
                    continue  # Noise
                indices = np.where(labels == label)[0]
                cluster_points = slice_points[indices]
                clusters.append({'points': cluster_points, 'slice': i})

        # Group clusters across slices
        boats = self.group_clusters_across_slices(clusters)

        return boats

    def group_clusters_across_slices(self, clusters):
        # Sort clusters by slice
        clusters.sort(key=lambda x: x['slice'])
        boat_clusters = []
        for cluster in clusters:
            added = False
            for boat in boat_clusters:
                # Check if cluster is close to any existing boat cluster
                distance = np.linalg.norm(cluster['points'][:, :2].mean(axis=0) - boat['points'][:, :2].mean(axis=0))
                if distance < 5.0:  # Adjust threshold as needed
                    # Merge clusters
                    boat['points'] = np.vstack((boat['points'], cluster['points']))
                    added = True
                    break
            if not added:
                # Start a new boat cluster
                boat_clusters.append({'points': cluster['points']})

        # Process boat clusters
        boats = []
        for boat_cluster in boat_clusters:
            boat_points = boat_cluster['points']
            # Apply size and shape filters
            if len(boat_points) < 50:
                continue

            x_min, x_max = boat_points[:, 0].min(), boat_points[:, 0].max()
            y_min, y_max = boat_points[:, 1].min(), boat_points[:, 1].max()
            z_min, z_max = boat_points[:, 2].min(), boat_points[:, 2].max()
            length = x_max - x_min
            width = y_max - y_min
            height = z_max - z_min

            # Aspect ratio and size filtering
            if width == 0:
                continue
            aspect_ratio = length / width

            if not (2.0 <= length <= 50.0 and 1.0 <= width <= 15.0 and 0.5 <= height <= 10.0):
                continue
            if not (0.5 <= aspect_ratio <= 20.0):
                continue

            centroid = np.mean(boat_points[:, :2], axis=0)
            boats.append({'points': boat_points, 'centroid': centroid})

        return boats

    def detect_boats_pca(self, points):
        # Initial clustering with adjusted eps
        clusterer = DBSCAN(eps=5.0, min_samples=10)
        labels = clusterer.fit_predict(points[:, :2])  # Use only x, y for clustering

        raw_boats = []
        for label in set(labels):
            if label == -1:
                continue  # Noise
            indices = np.where(labels == label)[0]
            cluster_points = points[indices]

            # Apply PCA
            pca = PCA(n_components=2)
            pca.fit(cluster_points[:, :2])  # Only use x, y for PCA
            eigenvalues = pca.explained_variance_

            # Check elongation
            if eigenvalues[0] / eigenvalues[1] > 3.0:
                # Cluster is elongated, likely a boat
                centroid = np.mean(cluster_points[:, :2], axis=0)
                length = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
                width = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
                raw_boats.append({'points': cluster_points, 'centroid': centroid, 'length': length, 'width': width})

        # After detecting raw boats, apply a more aggressive merging step
        merged_boats = self.merge_detections(raw_boats)
        return merged_boats

    def merge_detections(self, detections, distance_threshold=10.0):
        if not detections:
            return []

        merged = []
        centroids = np.array([d['centroid'] for d in detections])
        tree = cKDTree(centroids)

        processed = set()
        for i, detection in enumerate(detections):
            if i in processed:
                continue

            indices = tree.query_ball_point(detection['centroid'], distance_threshold)
            cluster = [j for j in indices if j not in processed]
            
            if cluster:
                # Merge all detections in this cluster
                all_points = np.vstack([detections[j]['points'] for j in cluster])
                merged_centroid = np.mean(all_points[:, :2], axis=0)
                merged.append({
                    'centroid': merged_centroid,
                    'points': all_points
                })
                processed.update(cluster)

        return merged

    def update_tracks(self, detected_boats):
        current_time = time.time()
        updated_boats = {}

        # Predict new positions for all existing tracks
        for boat_id, boat_info in self.tracked_boats.items():
            kf = boat_info['kf']
            kf.predict()

        # Associate detections with existing tracks
        unmatched_detections = []
        for detected_boat in detected_boats:
            centroid = detected_boat['centroid']
            matched = False

            for boat_id, boat_info in self.tracked_boats.items():
                kf = boat_info['kf']
                predicted_pos = np.array([kf.x[0], kf.x[2]])
                distance = np.linalg.norm(centroid - predicted_pos)

                if distance < self.tracking_threshold:
                    # Update existing track
                    kf.update(centroid)
                    updated_boats[boat_id] = {
                        'kf': kf,
                        'last_seen': current_time,
                        'points': detected_boat['points']
                    }
                    matched = True
                    print(f"Boat {boat_id} updated at coordinates: {centroid}")
                    break

            if not matched:
                unmatched_detections.append(detected_boat)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_id = self.next_boat_id
            self.next_boat_id += 1
            kf = self.create_kalman_filter(detection['centroid'])
            updated_boats[new_id] = {
                'kf': kf,
                'last_seen': current_time,
                'points': detection['points']
            }
            print(f"New boat {new_id} detected at coordinates: {detection['centroid']}")

        # Handle disappearing tracks
        for boat_id, boat_info in self.tracked_boats.items():
            if boat_id not in updated_boats:
                time_since_last_seen = current_time - boat_info['last_seen']
                if time_since_last_seen < self.max_disappeared_time:
                    # Retain the track for a while
                    updated_boats[boat_id] = {
                        'kf': boat_info['kf'],
                        'last_seen': boat_info['last_seen'],
                        'points': boat_info.get('points', np.empty((0, 3)))
                    }
                else:
                    print(f"Boat {boat_id} lost")

        self.tracked_boats = updated_boats

    def process_clusters(self, points, labels):
        boats = []
        for label in set(labels):
            if label == -1:
                continue  # Noise
            indices = np.where(labels == label)[0]
            boat_points = points[indices]
            # Filter out small clusters
            if len(boat_points) < 20:
                continue  # Skip clusters with fewer than 20 points

            # Compute the dimensions of the cluster
            x_min, x_max = boat_points[:, 0].min(), boat_points[:, 0].max()
            y_min, y_max = boat_points[:, 1].min(), boat_points[:, 1].max()
            length = x_max - x_min
            width = y_max - y_min

            # Aspect ratio (length to width ratio)
            if width == 0:
                continue  # Avoid division by zero
            aspect_ratio = length / width

            # Filter based on expected boat dimensions and aspect ratio
            min_length = 2.0  # Minimum boat length in meters
            max_length = 50.0  # Maximum boat length in meters
            min_width = 1.0   # Minimum boat width in meters
            max_width = 15.0  # Maximum boat width in meters
            min_aspect_ratio = 0.5  # Allow boats that are wider than they are long
            max_aspect_ratio = 20.0

            if not (min_length <= length <= max_length and min_width <= width <= max_width):
                continue  # Skip clusters that don't match expected boat sizes

            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                continue  # Skip clusters with unrealistic aspect ratios

            # Compute the centroid of the boat
            centroid = np.mean(boat_points[:, :2], axis=0)
            # Store the boat points and centroid
            boats.append({'points': boat_points, 'centroid': centroid})
        return boats

    def visualize(self, points):
        self.ax.clear()

        # Clamp Z values between 0 and 100 for the color representation
        z_values = np.clip(points[:, 2], 0, 100)
        
        # Color ALL points based on clamped altitude (Z-coordinate) and keep color scale constant
        sc = self.ax.scatter(points[:, 0], points[:, 1], c=z_values, s=1, cmap='viridis', vmin=0, vmax=100, label='Point Cloud')
        
        # Ensuring a proper colorbar is drawn for altitude with a fixed range
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            self.colorbar = plt.colorbar(sc, ax=self.ax, label='Altitude (meters)')
        else:
            self.colorbar.update_normal(sc)
        
        # Plot tracked boats
        for boat_id, boat_info in self.tracked_boats.items():
            kf = boat_info['kf']
            boat_points = boat_info.get('points', np.empty((0, 3)))
            predicted_position = np.array([kf.x[0], kf.x[2]])

            # Plot boat bounding box if points are available
            if boat_points.size > 0:
                x_min, x_max = boat_points[:, 0].min(), boat_points[:, 0].max()
                y_min, y_max = boat_points[:, 1].min(), boat_points[:, 1].max()
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height,
                                        linewidth=2, edgecolor='red', facecolor='none')
                self.ax.add_patch(rect)

            # Mark the predicted position
            self.ax.plot(predicted_position[0], predicted_position[1], 'ro', markersize=8)  # Red dot at the predicted position
            self.ax.text(predicted_position[0], predicted_position[1], f'ID {boat_id}', color='red', fontweight='bold', 
                         ha='center', va='bottom')  # Removed the bbox parameter

        # Re-apply axis settings
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Top-Down View of Detected Boats with Tracking')
        self.ax.grid(True)
        self.ax.legend(loc='upper right')

        plt.draw()
        plt.pause(0.001)  # This allows for continuous updating in a loop

    def destroy_node(self):
        plt.close(self.fig)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    boat_detector = BoatDetector()
    try:
        rclpy.spin(boat_detector)
    except KeyboardInterrupt:
        pass
    finally:
        boat_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
