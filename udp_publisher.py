import socket
import struct
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class UDPRelay(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'tracked_boats_bounding_boxes',
            self.callback,
            10)

    def callback(self, msg):
        # Pack data as little-endian floats
        data = struct.pack('<' + 'f' * len(msg.data), *msg.data)
        sock.sendto(data, (UDP_IP, UDP_PORT))

def main(args=None):
    rclpy.init(args=args)
    udp_relay = UDPRelay()
    try:
        rclpy.spin(udp_relay)
    except KeyboardInterrupt:
        pass
    finally:
        udp_relay.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
