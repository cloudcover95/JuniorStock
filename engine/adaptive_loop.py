# juniorstock/engine/adaptive_loop.py
import asyncio
import time
import mlx.core as mx
from juniorstock.engine.thermal_loop import ThermalExecutionEngine
from juniorstock.quant.learning.forward_ternary import TernaryForwardAdapter
from juniorstock.storage.journal.ring_journal import MmapRingJournal

class AdaptiveExecutionEngine(ThermalExecutionEngine):
    """
    Phase 21 Override: Integrates LFMP adaptation and Ring Journaling.
    The engine continuously perturbs the ATML weights to adapt to structural 
    drift in the market manifold without halting for backpropagation.
    """
    def __init__(self, node_id: str, base_target_hz: int = 100):
        super().__init__(node_id, base_target_hz)
        self.adapter = TernaryForwardAdapter()
        self.ring_journal = MmapRingJournal()
        
        # Establish base working memory for the predictive weights
        self.active_weights = mx.random.randint(-1, 2, (10, 10), dtype=mx.int8)

    async def execution_cycle(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] Adaptive Engine pinned to M4 Performance Cores.")
        
        asyncio.create_task(self._background_io_worker())

        while self.running:
            start_time = time.perf_counter()

            # 1. Evaluate baseline loss on current topological frame
            current_frame = mx.random.randint(-1, 2, (1, 10), dtype=mx.int8) # Simulated input
            target_frame = mx.random.randint(-1, 2, (1, 10), dtype=mx.int8)  # Simulated target truth
            
            # Base forward pass
            base_prediction = mx.matmul(current_frame, self.active_weights.T)
            base_loss = self.adapter.calculate_manifold_loss(base_prediction, target_frame)
            
            # 2. Generate Perturbation and evaluate perturbed loss
            perturbation = self.adapter.generate_ternary_perturbation(self.active_weights.shape)
            perturbed_weights = mx.clip(self.active_weights + perturbation, -1.0, 1.0)
            
            perturbed_prediction = mx.matmul(current_frame, perturbed_weights.T)
            perturbed_loss = self.adapter.calculate_manifold_loss(perturbed_prediction, target_frame)
            
            # 3. Apply Forward-Mode Ternary Update
            self.active_weights = self.adapter.compute_ternary_update(
                self.active_weights, base_loss, perturbed_loss, perturbation
            )
            
            # 4. Atomic Flush to Ring Journal
            self.ring_journal.append_state(self.active_weights)

            # Enforce cycle throttle and thermal boundaries
            current_soc_temp = self.smc_governor.read_soc_temperature()
            dynamic_hz = self.smc_governor.calculate_hz_modulation(current_soc_temp, self.base_hz)
            self.cycle_time = 1.0 / dynamic_hz

            elapsed = time.perf_counter() - start_time
            sleep_deficit = self.cycle_time - elapsed
            if sleep_deficit > 0:
                await asyncio.sleep(sleep_deficit)

    def shutdown(self):
        super().shutdown()
        self.ring_journal.shutdown()
        print("[JOURNAL GATE] Ring buffer flushed atomically.")
