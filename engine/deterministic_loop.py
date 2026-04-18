# juniorstock/engine/deterministic_loop.py
import asyncio
import time
import threading
import mlx.core as mx
from juniorstock.engine.zero_alloc_loop import ZeroAllocEngine
from juniorstock.hardware.mach.qos_router import AppleSiliconQoSRouter
from juniorstock.quant.forecasting.ternary_ar import TernaryAutoregressiveForecaster

class DeterministicExecutionEngine(ZeroAllocEngine):
    """
    Phase 17 Override: Injects Mach-level thread pinning to bypass Darwin scheduling jitter.
    Splits critical execution and I/O into deterministic core pipelines.
    """
    def __init__(self, node_id: str, target_hz: int = 100):
        super().__init__(node_id, target_hz)
        self.qos_router = AppleSiliconQoSRouter()
        self.taf_kernel = TernaryAutoregressiveForecaster(lookback_window=10)
        self.io_queue = asyncio.Queue()

    async def _background_io_worker(self):
        """
        Dedicated coroutine for Parquet flushing. Pinned to Efficiency Cores.
        """
        self.qos_router.pin_to_e_cores()
        print("[MACH ROUTE] I/O Worker pinned to M4 Efficiency Cores.")
        
        while self.running:
            tensor_state = await self.io_queue.get()
            try:
                # Execute Phase 15 emergency packing and disk flush
                await self.flush_state_to_disk(tensor_state)
            finally:
                self.io_queue.task_done()

    async def execution_cycle(self):
        # Hard pin the main high-frequency loop to Performance Cores
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] Execution Loop pinned to M4 Performance Cores.")
        
        # Ignite E-Core worker
        asyncio.create_task(self._background_io_worker())

        while self.running:
            start_time = time.perf_counter()

            # Dummy Time-Series generation for TAF testing
            ts_window = mx.random.normal((1, 10))
            
            # Predict Velocity
            velocity_signal = self.taf_kernel.forecast_velocity(ts_window)
            
            # Feature logic...
            if velocity_signal.item() != 0:
                # Offload state preservation to E-Core without blocking the P-Core execution loop
                if not self.io_queue.full():
                    self.io_queue.put_nowait(ts_window)

            # Strict Hz throttling
            elapsed = time.perf_counter() - start_time
            sleep_deficit = self.cycle_time - elapsed
            if sleep_deficit > 0:
                await asyncio.sleep(sleep_deficit)
