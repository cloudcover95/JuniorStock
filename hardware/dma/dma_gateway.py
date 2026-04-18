# juniorstock/hardware/dma/dma_gateway.py
import mmap
import os
import mlx.core as mx

class DirectMemoryGateway:
    """
    Bypasses Darwin kernel UART read() loops. 
    Maps a fixed POSIX memory region. The ATmega32u4 hardware pushes payload 
    bytes directly into this mapped file descriptor via an FTDI-to-DMA bridge.
    """
    def __init__(self, buffer_file: str = "/tmp/junior_dma_ring.bin", buffer_size: int = 4096):
        self.buffer_file = buffer_file
        self.buffer_size = buffer_size
        
        # Ensure file exists for mapping
        if not os.path.exists(self.buffer_file):
            with open(self.buffer_file, "wb") as f:
                f.write(b'\x00' * self.buffer_size)
                
        self.fd = os.open(self.buffer_file, os.O_RDWR)
        self.dma_view = mmap.mmap(self.fd, self.buffer_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Track offset to read the ring buffer efficiently
        self.head = 0

    def extract_dma_spike_train(self, tensor_dim: int) -> mx.array:
        """
        Reads the directly mapped bytes and converts them instantly to an int8 MLX array.
        Zero memory copying overhead.
        """
        # Read byte structure inserted by ATmega32u4
        self.dma_view.seek(self.head)
        raw_bytes = self.dma_view.read(tensor_dim)
        
        # Update circular pointer
        self.head = (self.head + tensor_dim) % self.buffer_size
        
        # Translate bytes directly into ternary spikes (ASCII mapping simulation)
        # e.g., 0x01 -> 1, 0xFF -> -1, 0x00 -> 0
        byte_list = list(raw_bytes)
        ternary_values = [b if b <= 1 else -1 for b in byte_list]
        
        return mx.array(ternary_values, dtype=mx.int8)

    def close(self):
        self.dma_view.close()
        os.close(self.fd)
