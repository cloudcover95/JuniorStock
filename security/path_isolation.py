# juniorstock/security/path_isolation.py
import os
import mlx.core as mx

class EnvironmentIsolationGate:
    """
    Strict runtime enforcement.
    System harvesters, packers, or AI agents must never traverse or ingest 
    from explicitly protected paths.
    """
    FORBIDDEN_PATHS = {"01_Legal", "02_Assets"}

    @classmethod
    def verify_io_path(cls, target_path: str):
        """
        Validates I/O execution paths before Parquet ingestion or SVD processing.
        """
        normalized_path = os.path.abspath(target_path)
        for forbidden in cls.FORBIDDEN_PATHS:
            if forbidden in normalized_path:
                # Immediate root node failure identification
                raise PermissionError(f"[SECURITY FAULT] Agent attempted traversal of forbidden path: {forbidden}")
        return True

    @classmethod
    def sanitize_tensor_memory(cls, memory_buffer: mx.array):
        """
        Zeroes out memory buffer if an anomalous traversal is detected mid-stream.
        """
        if memory_buffer is not None:
            memory_buffer = mx.zeros(memory_buffer.shape)
        return memory_buffer
