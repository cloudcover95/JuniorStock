# juniorstock/quant/hdam/hebbian_memory.py
import mlx.core as mx

class TernaryHebbianMemory:
    """
    O(1) single-shot associative memory matrix.
    Learns state-action mappings by binding ternary hypervectors via outer products.
    Bypasses gradient descent entirely, operating strictly within b1.58 limits.
    """
    def __init__(self, hyper_dim: int = 8192, action_dim: int = 16):
        self.d_x = hyper_dim
        self.d_y = action_dim
        # Pre-allocate zero-copy compatible uint8 buffer mapped to int8 space
        self.memory_matrix = mx.zeros((self.d_y, self.d_x), dtype=mx.int8)

    def associate_state_action(self, state_hv: mx.array, action_hv: mx.array):
        """
        Binds a systemic state hypervector to an optimal action vector.
        M_t+1 = clip(M_t + sgn(X outer Y), -1, 1)
        """
        # Outer product via batched matrix multiplication (reshaping to column/row vectors)
        # state_hv: (1, 8192) -> (8192, 1)
        # action_hv: (1, 16) -> (16, 1)
        
        state_col = mx.reshape(state_hv, (self.d_x, 1))
        action_row = mx.reshape(action_hv, (1, self.d_y))
        
        # Outer product
        outer_product = mx.matmul(action_row.T, state_col.T)
        
        # Ternary update logic
        hebbian_update = mx.astype(mx.sign(outer_product), mx.int8)
        
        # Element-wise addition and hard clamping back to {-1, 0, 1}
        updated_matrix = self.memory_matrix + hebbian_update
        self.memory_matrix = mx.clip(updated_matrix, -1, 1)

    def recall_action(self, state_hv: mx.array) -> mx.array:
        """
        Recalls the optimal action for a given systemic state.
        Y_hat = sgn(M * X)
        """
        # Dot product recall
        raw_recall = mx.matmul(self.memory_matrix, state_hv.T)
        
        # Signum projection back to ternary action bounds
        recalled_action = mx.clip(mx.sign(raw_recall), -1.0, 1.0)
        return mx.astype(recalled_action.T, mx.int8)
