# src/physiomanifold/tda_svd_core/koopman_market.py

import mlx.core as mx

class KoopmanOperatorDMD:
    """
    Extracts spatio-temporal market regimes from high-frequency tick data.
    Approximates the infinite-dimensional Koopman operator via Exact DMD.
    """
    def __init__(self, target_rank: int = 16):
        self.target_rank = target_rank

    def extract_dynamics(self, X_current: mx.array, X_next: mx.array) -> tuple:
        """
        Given market state sequences $X$ and $X'$, computes the reduced-order dynamics matrix $\tilde{A}$.
        $X' \approx A X \Rightarrow \tilde{A} = U^T A U = U^T X' V \Sigma^{-1}$
        """
        # 1. SVD on the current state mesh (MLX CPU stream for stability)
        U, S, Vt = mx.linalg.svd(X_current, stream=mx.cpu)
        
        # Truncate to target rank to eliminate market noise
        U_r = U[:, :self.target_rank]
        S_r = S[:self.target_rank]
        Vt_r = Vt[:self.target_rank, :]
        
        # 2. Compute the pseudo-inverse component: $V \Sigma^{-1}$
        # Add epsilon to diagonal to prevent zero-division
        S_inv = 1.0 / (S_r + 1e-9)
        
        # 3. Compute the reduced Koopman operator $\tilde{A}$
        # A_tilde = U_r^T * X_next * Vt_r^T * S_inv
        step_1 = mx.matmul(U_r.T, X_next)
        step_2 = mx.matmul(step_1, Vt_r.T)
        A_tilde = mx.matmul(step_2, mx.diag(S_inv))
        
        return A_tilde, U_r
