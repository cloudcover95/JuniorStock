# juniorstock/memory/mmap_tensor.py
import mmap
import posix_ipc
import numpy as np
import mlx.core as mx
import struct

class ZeroCopyTensorBuffer:
    """
    Bypasses standard OS network/socket stacks for inter-agent communication.
    Utilizes POSIX shared memory mapped directly into numpy/MLX buffers.
    Drastically reduces RAM duplication and system call overhead on Apple Silicon.
    """
    def __init__(self, shm_name: str, shape: tuple, dtype=np.float32):
        self.shm_name = f"/{shm_name}"
        self.shape = shape
        self.dtype = dtype
        self.bytes_size = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        
        # Internal state
        self._shm = None
        self._mmap = None
        self.tensor_view = None

    def initialize_buffer(self, create: bool = True):
        """
        Allocates or attaches to the POSIX shared memory block.
        """
        flags = posix_ipc.O_CREAT | posix_ipc.O_RDWR if create else posix_ipc.O_RDWR
        
        try:
            self._shm = posix_ipc.SharedMemory(self.shm_name, flags, size=self.bytes_size)
            self._mmap = mmap.mmap(self._shm.fd, self.shm_name.size)
            
            # Map a numpy view directly over the memory block (zero allocation)
            self.tensor_view = np.ndarray(self.shape, dtype=self.dtype, buffer=self._mmap)
            
            # Close the file descriptor, mmap maintains the memory lock
            posix_ipc.close_fd(self._shm.fd)
        except Exception as e:
            print(f"[MEMORY FAULT] IPC Allocation failed. Dimensions/Lock constraint: {e}")

    def write_tensor(self, mlx_tensor: mx.array):
        """
        Writes MLX output directly into the shared zero-copy buffer.
        """
        # Block array evaluation to synchronize MLX async graph
        mx.eval(mlx_tensor)
        # Direct numpy assignment via buffer memory view
        self.tensor_view[:] = np.array(mlx_tensor, copy=False)

    def read_tensor(self) -> mx.array:
        """
        Extracts shared memory block back into an MLX unified memory tensor.
        """
        return mx.array(self.tensor_view)

    def purge(self):
        """
        Unlinks and destroys the POSIX memory block to prevent OS-level leaks.
        """
        if self._mmap:
            self._mmap.close()
        if self._shm:
            try:
                posix_ipc.unlink_shared_memory(self.shm_name)
            except posix_ipc.ExistentialError:
                pass
