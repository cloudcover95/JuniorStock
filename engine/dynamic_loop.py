# juniorstock/engine/dynamic_loop.py
import asyncio
import time
import mlx.core as mx
from juniorstock.engine.cognitive_loop import CognitiveExecutionEngine
from juniorstock.nodes.ingestion.l2_compressor import L2DeltaCompressor

class DynamicExecutionEngine(CognitiveExecutionEngine):
    """
    Phase 19 Override: Integrates the L2 compressor to filter incoming data.
    Only processes ticks that mathematically deform the Betti manifold.
    """
    def __init__(self, node_id: str, target_hz: int = 100):
        super().__init__(node_id, target_hz)
        self.l2_compressor = L2DeltaCompressor(tensor_dim=10)

    async def execution_cycle(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] Dynamic Engine pinned to M4 Performance Cores.")
        
        asyncio.create_task(self._background_io_worker())

        while self.running:
            start_time = time.perf_counter()

            # Simulate heavy L2 REST/WS ingestion array
            raw_l2_ingestion = mx.random.normal((1, 10))
            
            # Hardware-level topological gating
            manifold_shift_detected, compressed_tick = self.l2_compressor.process_tick(raw_l2_ingestion)
            
            if manifold_shift_detected:
                # --------------------------------------------------
                # Execute core algorithmic cascade ONLY on valid delta
                # --------------------------------------------------
                velocity_signal = self.taf_kernel.forecast_velocity(compressed_tick)
                
                if mx.abs(velocity_signal).item() == 1:
                    historical_indices = self.memory_palace.retrieve_isomorphic_state(compressed_tick)
                    # Router dispatch logic...
                    
                self.memory_palace.ingest_historical_manifold(compressed_tick)
            else:
                # Bypass complete execution tree. Idle loop saves power draw.
                pass

            elapsed = time.perf_counter() - start_time
            sleep_deficit = self.cycle_time - elapsed
            if sleep_deficit > 0:
                await asyncio.sleep(sleep_deficit)
