# juniorstock/memory/cache/l1_aligner.py
import mlx.core as mx

class L1CacheAligner:
    """
    Forces MLX tensors to align perfectly with Apple Silicon 128-byte L1 cache lines.
    Eliminates cache-miss penalties and prefetcher stalling during SIMD LUT synthesis.
    """
    def __init__(self, dtype=mx.int8):
        self.cache_line_bytes = 128
        self.dtype = dtype
        # Calculate elements per cache line
        self.elements_per_line = self.cache_line_bytes // mx.array([0], dtype=self.dtype).nbytes

    def pad_to_l1(self, tensor: mx.array) -> tuple:
        """
        Pads the innermost dimension to a multiple of the cache line size.
        Returns the aligned tensor and original shape for zero-padding truncation during extraction.
        """
        original_shape = tensor.shape
        inner_dim = original_shape[-1]
        
        remainder = inner_dim % self.elements_per_line
        if remainder == 0:
            return tensor, original_shape
            
        pad_size = self.elements_per_line - remainder
        pad_shape = list(original_shape)
        pad_shape[-1] = pad_size
        
        # Zero padding is mathematically neutral in b1.58 additive logic
        padding = mx.zeros(pad_shape, dtype=self.dtype)
        aligned_tensor = mx.concatenate([tensor, padding], axis=-1)
        
        return aligned_tensor, original_shape

    def strip_l1_padding(self, aligned_tensor: mx.array, original_shape: tuple) -> mx.array:
        """
        Slices the cache-line padding off before transmitting across the AX mesh.
        """
        return aligned_tensor[..., :original_shape[-1]]
