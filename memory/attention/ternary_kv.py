# juniorstock/memory/attention/ternary_kv.py
import mlx.core as mx
from juniorstock.quant.bitnet.quantize import BitNetQuantizer

class CompressedTernaryAttention:
    """
    Replaces continuous softmax logic with a b1.58 ternary topological state matrix.
    Eliminates fp16 exp() calls, locking SoC power consumption across long inference sequences.
    """
    def __init__(self, sequence_length: int, hidden_dim: int):
        self.seq_len = sequence_length
        self.dim = hidden_dim
        self.quantizer = BitNetQuantizer()
        
        # Pre-allocate zeroed int8 buffers for Zero-Copy updates
        self.k_cache = mx.zeros((self.seq_len, self.dim), dtype=mx.int8)
        self.v_cache = mx.zeros((self.seq_len, self.dim), dtype=mx.int8)
        self.cache_idx = 0

    def _quantize_projection(self, tensor: mx.array) -> mx.array:
        """
        Forces Q, K, V projections into packed int8 ternary states.
        """
        q_tensor, _ = self.quantizer.quantize_activations(tensor, bits=2)
        return mx.astype(q_tensor, mx.int8)

    def forward(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """
        Executes the BitNet Attention equivalent.
        """
        # Compress incoming projections
        q_ternary = self._quantize_projection(q)
        k_ternary = self._quantize_projection(k)
        v_ternary = self._quantize_projection(v)

        # Update cyclic cache
        if self.cache_idx < self.seq_len:
            self.k_cache[self.cache_idx] = k_ternary
            self.v_cache[self.cache_idx] = v_ternary
            self.cache_idx += 1
            
        # Extract active cache sequence
        k_active = self.k_cache[:self.cache_idx]
        v_active = self.v_cache[:self.cache_idx]

        # Multiplier-free scaled attention (Simulated native bind integration)
        # Q(K^T) utilizing native matrix multiplication (or custom mlx_ternary_ext)
        raw_scores = mx.matmul(q_ternary, k_active.T)
        
        # Scale and quantize the attention weights (Bypassing Softmax entirely)
        gamma_qk = mx.mean(mx.abs(raw_scores)) + 1e-9
        attention_weights = mx.clip(mx.round(raw_scores / gamma_qk), -1.0, 1.0)
        
        # Final accumulation Z = A * V
        z_out = mx.matmul(mx.astype(attention_weights, mx.int8), v_active)
        
        return mx.astype(z_out, mx.float16) # Return to fp16 for residual stream
