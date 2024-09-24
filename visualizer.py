import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        self.subscription = self.create_subscription(
            PointCloud2,
            'lidar_pointcloud',
            self.listener_callback,
            10)

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("LiDAR Point Cloud")
        self.point_cloud = o3d.geometry.PointCloud()
        self.is_first_frame = True

    def listener_callback(self, msg):
        points = self.pointcloud2_to_array(msg)
        self.visualize(points)

    def pointcloud2_to_array(self, cloud_msg):
        cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
        points = cloud_arr.reshape(-1, 3)
        return points

    def visualize(self, points):
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        if self.is_first_frame:
            self.vis.add_geometry(self.point_cloud)
            self.is_first_frame = False
        else:
            self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy_node(self):
        self.vis.destroy_window()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    lidar_visualizer = LidarVisualizer()
    try:
        rclpy.spin(lidar_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
