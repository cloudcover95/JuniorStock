# juniorstock/quant/bitnet/manifold_compressor.py
import mlx.core as mx
import mlx.nn as nn
from juniorstock.quant.bitnet.linear import BitNetLinearLayer

class TernaryAutoencoder(nn.Module):
    """
    Compresses high-dimensional market manifolds into b1.58 ternary states 
    for ultra-low bandwidth local network transmission.
    Replaces heavy SVD projections with direct MLX/Metal ternary neural mappings.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = BitNetLinearLayer(input_dim, bottleneck_dim)
        self.decoder = BitNetLinearLayer(bottleneck_dim, input_dim)

    def compress_for_broadcast(self, manifold: mx.array) -> mx.array:
        """
        Encodes the input tensor directly to a ternary representation.
        Returns int8 packed array for UDP payload minimization.
        """
        # Metal-accelerated BitNet linear pass
        encoded = self.encoder(manifold)
        
        # Clamp strictly to ternary bounds and cast to smallest int type
        ternary_state = mx.clip(mx.round(encoded), -1.0, 1.0)
        return mx.astype(ternary_state, mx.int8)

    def decompress_from_stream(self, ternary_payload: mx.array) -> mx.array:
        """
        Reconstructs the approximate FP16 manifold on the receiving node.
        """
        # Cast back to continuous space for the decoder layer
        fp_payload = mx.astype(ternary_payload, mx.float16)
        return self.decoder(fp_payload)
