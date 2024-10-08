import socket
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on UDP {UDP_IP}:{UDP_PORT}")

while True:
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    print(f"Received message from {addr}: {data}")
    
    # Unpack the float data
    num_floats = len(data) // 4  # Since each float is 4 bytes (32 bits)
    floats = struct.unpack('<' + 'f' * num_floats, data)
    print(f"Received floats: {floats}")
