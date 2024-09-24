import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TwistStamped, QuaternionStamped
from ouster.sdk import open_source, client, pcap
import numpy as np
import threading
import sys
import select
import termios
import tty
import time
import math

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'lidar_pointcloud', 10)
        self.speed_publisher = self.create_publisher(TwistStamped, 'boat_speed', 10)
        self.orientation_publisher = self.create_publisher(QuaternionStamped, 'boat_orientation', 10)
        self.timer = self.create_timer(0.1, self.publish_data)  # 10Hz publish rate

        # Choose source of lidar data
        self.RECORDING = True
        if self.RECORDING:
            pcap_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/boatpassing.pcap'
            metadata_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/boatpassing.json'
            with open(metadata_path, 'r') as f:
                info = client.SensorInfo(f.read())
            self.source = pcap.Pcap(pcap_path, info)
            self.scans = client.Scans(self.source)
        else:
            sensor_url = '<SENSOR-HOSTNAME-OR-IP>'
            self.source = open_source(sensor_url)
            self.scans = client.Scans(self.source)

        self.scan_iter = iter(self.scans)
        self.xyz_lut = client.XYZLut(info)

        # Pause and step features
        self.paused = False
        self.step_forward = False
        self.step_backward = False
        self.scans_buffer = []  # Buffer to store scans for backward navigation
        self.current_scan_index = -1  # Index of the current scan in the buffer

        self.input_thread = threading.Thread(target=self.input_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

        # Save terminal settings to restore later
        self.orig_settings = termios.tcgetattr(sys.stdin)

        # Simulation variables for speed and orientation
        self.start_time = time.time()
        self.last_publish_time = self.start_time
        self.current_speed = 0.0  # meters per second
        self.current_yaw = 0.0    # radians
        self.speed_increment = 0.1  # m/s per second
        self.yaw_rate = 0.01       # radians per second

    def input_listener(self):
        tty.setcbreak(sys.stdin)
        try:
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == ' ':
                        self.paused = not self.paused
                        state = 'paused' if self.paused else 'running'
                        self.get_logger().info(f'Playback {state}. Press space bar to toggle.')
                    elif key.lower() == 'n':
                        if self.paused:
                            self.step_forward = True
                    elif key.lower() == 'p':
                        if self.paused:
                            self.step_backward = True
                else:
                    time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f'Input listener error: {e}')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)

    def publish_data(self):
        if self.paused:
            if self.step_forward:
                self.step_forward = False
                self.advance_frame()
            elif self.step_backward:
                self.step_backward = False
                self.go_back_frame()
            else:
                # Still publish speed and orientation when paused
                self.publish_speed_and_orientation()
            return
        else:
            self.advance_frame()
            self.publish_speed_and_orientation()

    def advance_frame(self):
        try:
            if self.current_scan_index < len(self.scans_buffer) - 1:
                # Move forward within the buffer
                self.current_scan_index += 1
                scan = self.scans_buffer[self.current_scan_index]
                self.publish_scan(scan)
                self.get_logger().info(f'Advancing to frame {self.current_scan_index}')
            else:
                # Read a new scan from the iterator
                scan = next(self.scan_iter)
                self.scans_buffer.append(scan)
                self.current_scan_index += 1
                self.publish_scan(scan)
                self.get_logger().info(f'Reading new scan, frame {self.current_scan_index}')
        except StopIteration:
            self.get_logger().info('Reached end of pcap file. Shutting down...')
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Error advancing frame: {e}')

    def go_back_frame(self):
        if self.current_scan_index > 0:
            self.current_scan_index -= 1
            scan = self.scans_buffer[self.current_scan_index]
            self.publish_scan(scan)
            self.get_logger().info(f'Moved back to frame {self.current_scan_index}')
        else:
            self.get_logger().info('Already at the first frame.')

    def publish_scan(self, scan):
        try:
            # Extract the first timestamp from the scan (in nanoseconds)
            timestamp_ns = scan.timestamp[0]  # Use the first timestamp in the array

            # Convert the timestamp from nanoseconds to seconds and nanoseconds for ROS
            timestamp_sec = int(timestamp_ns // 1_000_000_000)  # Seconds as integer
            timestamp_nsec = int(timestamp_ns % 1_000_000_000)  # Nanoseconds as integer

            # Calculate dt (time difference) from the previous scan
            if hasattr(self, 'previous_timestamp'):
                dt = (timestamp_ns - self.previous_timestamp) / 1e9  # Convert nanoseconds to seconds
                self.previous_timestamp = timestamp_ns
            else:
                dt = 0.1  # Fallback to a default time step if it's the first scan
                self.previous_timestamp = timestamp_ns

            # Process the XYZ data from the point cloud
            xyz = self.xyz_lut(scan)
            xyz = xyz.reshape(-1, 3)

            # Create PointCloud2 message
            msg = PointCloud2()
            msg.header.stamp.sec = timestamp_sec  # Set seconds part of the timestamp (integer)
            msg.header.stamp.nanosec = timestamp_nsec  # Set nanoseconds part of the timestamp (integer)
            msg.header.frame_id = 'lidar_frame'
            msg.height = 1
            msg.width = xyz.shape[0]
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            msg.is_bigendian = False
            msg.point_step = 12
            msg.row_step = msg.point_step * msg.width
            msg.is_dense = True
            msg.data = xyz.astype(np.float32).tobytes()

            # Publish the point cloud data with the correct timestamp
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing pointcloud, frame_id: {scan.frame_id}, index: {self.current_scan_index}')

            # Log the dt to verify the time difference
            self.get_logger().info(f'Time difference (dt) between scans: {dt:.6f} seconds')

        except Exception as e:
            self.get_logger().error(f'Error publishing scan: {e}')


    def publish_speed_and_orientation(self):
        try:
            current_time = time.time()
            dt = current_time - self.last_publish_time
            self.last_publish_time = current_time

            # Simulate speed change
            # self.current_speed += self.speed_increment * dt  # m/s
            self.current_speed = 0.0 # m/s

            # Simulate yaw change
            # self.current_yaw += self.yaw_rate * dt  # radians
            self.current_yaw = 0.0 # radians

            # Normalize yaw to [-pi, pi]
            self.current_yaw = (self.current_yaw + math.pi) % (2 * math.pi) - math.pi

            # Publish speed
            speed_msg = TwistStamped()
            speed_msg.header.stamp = self.get_clock().now().to_msg()
            speed_msg.header.frame_id = 'lidar_frame'
            speed_msg.twist.linear.x = self.current_speed * math.cos(self.current_yaw)
            speed_msg.twist.linear.y = self.current_speed * math.sin(self.current_yaw)
            speed_msg.twist.linear.z = 0.0
            speed_msg.twist.angular.x = 0.0
            speed_msg.twist.angular.y = 0.0
            speed_msg.twist.angular.z = self.yaw_rate

            self.speed_publisher.publish(speed_msg)

            # Publish orientation
            orientation_msg = QuaternionStamped()
            orientation_msg.header.stamp = self.get_clock().now().to_msg()
            orientation_msg.header.frame_id = 'lidar_frame'
            quaternion = self.yaw_to_quaternion(self.current_yaw)
            orientation_msg.quaternion = quaternion

            self.orientation_publisher.publish(orientation_msg)

            self.get_logger().info(f'Publishing speed: {self.current_speed:.2f} m/s, yaw: {self.current_yaw:.2f} rad')

        except Exception as e:
            self.get_logger().error(f'Error publishing speed and orientation: {e}')

    def yaw_to_quaternion(self, yaw):
        """Converts a yaw angle (in radians) to a quaternion."""
        quaternion = QuaternionStamped().quaternion
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion


    def destroy_node(self):
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
        super().destroy_node()

    def __del__(self):
        # Ensure terminal settings are restored when the object is deleted
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)

def main(args=None):
    rclpy.init(args=args)
    lidar_publisher = LidarPublisher()
    try:
        rclpy.spin(lidar_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
