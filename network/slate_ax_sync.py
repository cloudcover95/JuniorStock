# juniorstock/network/slate_ax_sync.py
import socket
import struct
import mlx.core as mx
import numpy as np

class SlateAXMeshRouter:
    """
    Zero-dependency UDP multicast router.
    Broadcasts compressed int8 ternary states across the local subnet.
    Optimized for multi-agent synchronization (M4 Host -> Orange Pi Nodes).
    """
    def __init__(self, multicast_group: str = '224.0.0.1', port: int = 10000):
        self.multicast_group = multicast_group
        self.port = port
        
        # Socket configuration for local mesh broadcasting
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        
        # Receiver socket binding
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind(('', self.port))
        
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        self.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.recv_sock.setblocking(False)

    def broadcast_ternary_state(self, node_id: str, ternary_array: mx.array):
        """
        Serializes the int8 MLX tensor to bytes and broadcasts to the AX subnet.
        """
        # Synchronize MLX graph before extracting values
        mx.eval(ternary_array)
        
        # Zero-copy cast to numpy for byte serialization
        np_arr = np.array(ternary_array, copy=False)
        shape_bytes = struct.pack('II', np_arr.shape[0], np_arr.shape[1])
        payload = node_id.encode('utf-8').ljust(16, b'\0') + shape_bytes + np_arr.tobytes()
        
        try:
            self.sock.sendto(payload, (self.multicast_group, self.port))
        except OSError as e:
            print(f"[AX ROUTER FAULT] Multicast dispatch failed. ERR: {e}")

    def poll_mesh_updates(self) -> list:
        """
        Non-blocking poll for incoming ternary manifolds from peer agents.
        Returns a list of decoded (node_id, mx.array) tuples.
        """
        updates = []
        while True:
            try:
                data, addr = self.recv_sock.recvfrom(4096)
                node_id = data[:16].decode('utf-8').strip('\0')
                rows, cols = struct.unpack('II', data[16:24])
                
                # Reconstruct MLX tensor directly from byte buffer
                np_arr = np.frombuffer(data[24:], dtype=np.int8).reshape((rows, cols))
                tensor = mx.array(np_arr)
                
                updates.append((node_id, tensor))
            except BlockingIOError:
                break # Queue empty
            except Exception as e:
                print(f"[AX ROUTER FAULT] Mesh ingestion error. ERR: {e}")
                break
                
        return updates
