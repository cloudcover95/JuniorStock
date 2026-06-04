# path: src/hardware/backend_router.py

from enum import Enum
from typing import Any


class Backend(Enum):
    MLX = "mlx"
    CUDA = "cuda"


class BackendRouter:
    """
    Hardware-agnostic backend router.
    Used to route BitNet/ternary workloads across available silicon.
    """

    def __init__(self):
        self._current = Backend.MLX
        self._matmul_impl = None
        self.set_backend(Backend.MLX)

    def set_backend(self, backend: Backend):
        self._current = backend
        if backend == Backend.MLX:
            from ..bitnet.kernels.ternary_kernels import ternary_matmul as impl
        elif backend == Backend.CUDA:
            from ..bitnet.kernels.cuda_kernels import cuda_ternary_matmul as impl
        else:
            raise ValueError("Unsupported backend")
        self._matmul_impl = impl

    def ternary_matmul(self, input: Any, ternary_weight: Any, scale: Any):
        return self._matmul_impl(input, ternary_weight, scale)
