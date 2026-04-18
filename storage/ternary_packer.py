# juniorstock/storage/ternary_packer.py
import mlx.core as mx

class TernaryBitPacker:
    """
    Vectorized 3^5 bit-packing for b1.58 ternary arrays.
    Compresses 5 ternary values (-1, 0, 1) into a single uint8 byte.
    Bypasses standard float16/int8 disk I/O bottlenecks.
    """
    def __init__(self):
        # Powers of 3 vector for rapid dot product mapping
        self.powers_of_3 = mx.array([1, 3, 9, 27, 81], dtype=mx.uint32)

    def pack_tensor(self, ternary_tensor: mx.array) -> tuple:
        """
        Packs a flat ternary MLX array into a compressed uint8 array.
        Returns the packed array and the original shape for reconstruction.
        """
        original_shape = ternary_tensor.shape
        flat_tensor = mx.flatten(ternary_tensor)
        
        # Calculate padding required to make length a multiple of 5
        remainder = flat_tensor.size % 5
        if remainder != 0:
            pad_size = 5 - remainder
            # Pad with zeros (maps to 1 in the 0-2 space, safe for decoding)
            padding = mx.zeros((pad_size,), dtype=flat_tensor.dtype)
            flat_tensor = mx.concatenate([flat_tensor, padding])

        # Map domain {-1, 0, 1} -> {0, 1, 2}
        mapped_tensor = mx.astype(flat_tensor + 1, mx.uint32)
        
        # Reshape to (N, 5) and calculate dot product with powers of 3
        reshaped = mx.reshape(mapped_tensor, (-1, 5))
        packed = mx.sum(reshaped * self.powers_of_3, axis=1)
        
        return mx.astype(packed, mx.uint8), original_shape

    def unpack_tensor(self, packed_tensor: mx.array, original_shape: tuple) -> mx.array:
        """
        Reconstructs the original {-1, 0, 1} ternary tensor from uint8 space.
        """
        packed_32 = mx.astype(packed_tensor, mx.uint32)
        
        # Vectorized modulo arithmetic for base-3 decoding
        unpacked_mapped = mx.stack([
            (packed_32 // self.powers_of_3[0]) % 3,
            (packed_32 // self.powers_of_3[1]) % 3,
            (packed_32 // self.powers_of_3[2]) % 3,
            (packed_32 // self.powers_of_3[3]) % 3,
            (packed_32 // self.powers_of_3[4]) % 3
        ], axis=1)
        
        # Flatten, map domain {0, 1, 2} -> {-1, 0, 1}, and truncate padding
        flat_unpacked = mx.flatten(unpacked_mapped)
        original_size = mx.prod(mx.array(original_shape)).item()
        truncated = flat_unpacked[:original_size]
        
        # Return restored shape and type
        return mx.astype(mx.reshape(truncated, original_shape) - 1, mx.int8)
