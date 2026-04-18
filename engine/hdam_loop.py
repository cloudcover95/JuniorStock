# juniorstock/engine/hdam_loop.py
import asyncio
import mlx.core as mx
from juniorstock.engine.entropy_loop import EntropyExecutionEngine
from juniorstock.quant.hdam.hebbian_memory import TernaryHebbianMemory
from juniorstock.execution.phy.phy_injector import PHYNetworkInjector

class HDAMExecutionEngine(EntropyExecutionEngine):
    """
    Phase 23 Override: Integrates Hebbian single-shot learning and PHY bypass.
    Execution is mapped via associative memory and fired entirely outside the Darwin kernel.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id) 
        self.hdam = TernaryHebbianMemory(hyper_dim=8192, action_dim=16)
        self.phy_injector = PHYNetworkInjector()

    async def execution_cycle(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] HDAM Engine pinned to M4 P-Cores. OS Network Stack Bypassed.")
        
        asyncio.create_task(self._background_io_worker())

        while self.running:
            # 1. Thermal Audit
            current_soc_temp = self.smc_governor.read_soc_temperature()
            if current_soc_temp >= self.smc_governor.critical_temp_c:
                await asyncio.sleep(5.0)
                continue

            # 2. Ingest and Compress
            raw_l2_ingestion = mx.random.normal((10, 10))
            manifold_shift_detected, compressed_tick = self.l2_compressor.process_tick(raw_l2_ingestion)

            if manifold_shift_detected:
                # 3. Entropy Gate
                if self.entropy_gate.evaluate_manifold_chaos(compressed_tick):
                    
                    # Bind into High-Dimensional Space
                    velocity_signal = self.taf_kernel.forecast_velocity(compressed_tick)
                    hv_asset_a = self.hdc.project_and_bind("XAU_USD", velocity_signal[0].item())
                    systemic_hv = self.hdc.bundle_hypervectors([hv_asset_a])
                    
                    # 4. O(1) Recall via Hebbian Associative Memory
                    action_vector = self.hdam.recall_action(systemic_hv)
                    
                    # Decode action vector (simplified feature score check)
                    if mx.sum(action_vector).item() > 5:
                        
                        # --- PHY INJECTION ---
                        # Pre-compiled payload bypassed directly to hardware
                        dummy_payload = "0x" + "FF" * 32
                        self.phy_injector.inject_raw_frame(dummy_payload)
                        
                        # Apply LFMP or single-shot Hebbian update post-execution based on validation oracle
                        # self.hdam.associate_state_action(systemic_hv, optimal_action_hv)

                    self.ring_journal.append_state(action_vector)

            await asyncio.sleep(0)
