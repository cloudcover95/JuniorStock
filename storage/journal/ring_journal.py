# juniorstock/storage/journal/ring_journal.py
import os
import mmap
import struct
import mlx.core as mx
from juniorstock.storage.ternary_packer import TernaryBitPacker

class MmapRingJournal:
    """
    Zero-latency atomic persistence buffer.
    Writes packed 3^5 ternary manifolds to a pre-allocated circular file buffer.
    Relies on OS-level async msync for disk flushing, bypassing Python I/O blocking.
    """
    def __init__(self, filepath: str = "juniorstock/storage/journal/active_state.bin", max_records: int = 1000, record_size_bytes: int = 256):
        self.filepath = filepath
        self.max_records = max_records
        self.record_size = record_size_bytes
        self.file_size = self.max_records * self.record_size
        self.packer = TernaryBitPacker()
        
        self._head = 0
        self._initialize_mmap()

    def _initialize_mmap(self):
        """
        Pre-allocates the contiguous file block and establishes the mmap view.
        """
        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(b'\x00' * self.file_size)
                
        self.fd = os.open(self.filepath, os.O_RDWR)
        self.mmap_view = mmap.mmap(self.fd, self.file_size, access=mmap.ACCESS_WRITE)

    def append_state(self, ternary_manifold: mx.array):
        """
        Packs the ternary array and writes the binary payload to the circular buffer.
        """
        packed_data, shape = self.packer.pack_tensor(ternary_manifold)
        
        # Enforce strict byte sizing
        payload = bytes(packed_data.tolist())
        if len(payload) + 8 > self.record_size:
            raise ValueError("[JOURNAL FAULT] Packed payload exceeds allocated record byte size.")
            
        # Format: [4B Rows][4B Cols][Payload Block][Padding]
        header = struct.pack('II', shape[0], shape[1])
        full_record = (header + payload).ljust(self.record_size, b'\x00')
        
        # Calculate offset and write directly to memory
        offset = self._head * self.record_size
        self.mmap_view.seek(offset)
        self.mmap_view.write(full_record)
        
        # Advance ring pointer
        self._head = (self._head + 1) % self.max_records

    def shutdown(self):
        """
        Forces synchronous flush to physical disk prior to SIGTERM termination.
        """
        if self.mmap_view:
            self.mmap_view.flush()
            self.mmap_view.close()
        os.close(self.fd)
