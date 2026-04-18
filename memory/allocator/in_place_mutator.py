# juniorstock/memory/allocator/in_place_mutator.py
import numpy as np
import mlx.core as mx
from juniorstock.memory.mmap_tensor import ZeroCopyTensorBuffer

class TernaryStateMutator:
    """
    Bypasses MLX array immutability to enforce zero-allocation updates.
    Operates directly on the underlying POSIX mmap buffer view via NumPy.
    Used for instantaneous bit-flips during macro-automata hardware triggers.
    """
    def __init__(self, buffer_ref: ZeroCopyTensorBuffer):
        self.buffer = buffer_ref
        if self.buffer.tensor_view is None:
            raise MemoryError("[MUTATOR FAULT] POSIX mmap view not initialized.")

    def bit_flip_state(self, row_idx: int, col_idx: int):
        """
        Direct memory register mutation. Flips b1.58 states (-1 -> 1, 1 -> -1)
        without triggering the MLX allocator or Python Garbage Collector.
        """
        current_val = self.buffer.tensor_view[row_idx, col_idx]
        
        # Fast bitwise-like flip for ternary bound values
        if current_val == 1:
            self.buffer.tensor_view[row_idx, col_idx] = -1
        elif current_val == -1:
            self.buffer.tensor_view[row_idx, col_idx] = 1
            
        # 0 remains 0 (dead node)

    def force_zero_manifold(self):
        """
        Instantaneous memory zeroing for emergency load-shedding states.
        O(1) allocation cost.
        """
        self.buffer.tensor_view.fill(0)
