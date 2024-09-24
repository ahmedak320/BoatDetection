import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from ouster.sdk import open_source, client, pcap
import numpy as np
import threading
import sys
import select
import termios
import tty
import time

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'lidar_pointcloud', 10)
        self.timer = self.create_timer(0.1, self.publish_pointcloud)  # 10Hz publish rate

        # Choose source of lidar data
        self.RECORDING = True
        if self.RECORDING:
            pcap_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/boatpassing.pcap'
            metadata_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/boatpassing.json'
            # pcap_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/festivalcity.pcap'
            # metadata_path = '/home/ahmed/Desktop/new_pc_code/Ouster_Data/festivalcity.json'
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

    def publish_pointcloud(self):
        if self.paused:
            if self.step_forward:
                self.step_forward = False
                self.advance_frame()
            elif self.step_backward:
                self.step_backward = False
                self.go_back_frame()
            return
        else:
            self.advance_frame()

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
            xyz = self.xyz_lut(scan)
            xyz = xyz.reshape(-1, 3)

            # Create PointCloud2 message
            msg = PointCloud2()
            msg.header.stamp = self.get_clock().now().to_msg()
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

            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing pointcloud, frame_id: {scan.frame_id}, index: {self.current_scan_index}')
        except Exception as e:
            self.get_logger().error(f'Error publishing scan: {e}')

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
