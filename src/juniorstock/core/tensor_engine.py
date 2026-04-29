import numpy as np

class KPZKinematicEngine:
    def __init__(self, nu=1.0, lambda_=2.0, eta=0.01):
        self.nu, self.lambda_, self.eta = nu, lambda_, eta

    def compute_k_alpha(self, Z_tensor: np.ndarray) -> np.ndarray:
        returns = np.diff(Z_tensor, axis=1)
        local_eta = np.std(returns[:, -10:], axis=1) + self.eta
        manifold_mean = np.mean(Z_tensor[:, -1])
        laplacian = self.nu * np.abs(Z_tensor[:, -1] - manifold_mean)
        nonlinear_growth = (self.lambda_ / 2.0) * (returns[:, -1] ** 2)
        return np.round(nonlinear_growth / (laplacian + local_eta + 1e-8), 4)

class UnifiedFinancialTensor:
    def __init__(self, h_bar_mkt=0.01):
        self.h_bar_mkt = h_bar_mkt
        self._prev_q = None
        self.kpz = KPZKinematicEngine()

    def process_manifold(self, C: np.ndarray, H: np.ndarray, L: np.ndarray):
        if C.ndim == 1: C = C.reshape(1, -1)
        N, T = C.shape
        means = np.mean(C, axis=1, keepdims=True)
        stds = np.maximum(np.std(C, axis=1, keepdims=True), 1e-8)
        current_spots = C[:, -1]
        
        Z_scores = (C - means) / stds
        returns = np.diff(C, axis=1) / np.maximum(C[:, :-1], 1e-8)
        base_vols = np.std(returns, axis=1)
        recent_deltas = np.std(returns[:, -10:], axis=1) if T > 10 else base_vols
        
        q_marks = 1.0 - np.exp(-np.abs(Z_scores[:, -1]) * (recent_deltas / np.maximum(base_vols, self.h_bar_mkt)))
        turtle_alignment = (current_spots - L[:, -1]) / (H[:, -1] - L[:, -1] + 1e-8)
        k_alphas = self.kpz.compute_k_alpha(Z_scores)
        
        return {
            "spot": current_spots, "z_score": Z_scores[:, -1], "full_z_history": Z_scores,
            "q_mark": q_marks, "turtle_alignment": turtle_alignment, "k_alpha": k_alphas
        }

    def compute_delta_q(self, current_q: np.ndarray) -> np.ndarray:
        if self._prev_q is None:
            self._prev_q = current_q
            return np.zeros_like(current_q)
        delta = current_q - self._prev_q
        self._prev_q = current_q
        return delta
