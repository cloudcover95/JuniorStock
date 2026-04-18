# juniorstock/engine/spiking_loop.py
import signal
import mlx.core as mx
from juniorstock.engine.interrupt_loop import InterruptExecutionEngine
from juniorstock.hardware.dma.dma_gateway import DirectMemoryGateway

class SpikingExecutionEngine(InterruptExecutionEngine):
    """
    Phase 26 Override: Replaces discrete tensor routing with Continuous Spiking Automata.
    Wakes via SIGIO, ingests zero-copy DMA memory, and routes via Metal LIF kernels.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.dma_gate = DirectMemoryGateway()
        
        # LIF Network Topology initialization
        self.num_neurons = 128
        self.tensor_dim = 10
        self.lif_weights = mx.random.randint(-1, 2, (self.num_neurons, self.tensor_dim), dtype=mx.int8)
        self.membrane_potentials = mx.zeros((self.num_neurons,), dtype=mx.int32)
        
        # LIF Parameters
        self.beta_decay = 4 # Equivalent to right bit-shift by 4 (approx 0.93 leakage)
        self.spike_threshold = 15

    def execution_cycle_blocking(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] ASTA Engine pinned. Waiting for DMA Interrupts.")
        
        self.interrupt_gate.arm_hardware_interrupt()

        while self.running:
            self.interrupt_gate.interrupt_triggered = False
            signal.pause() 
            
            if not self.interrupt_gate.interrupt_triggered:
                continue

            tick_start_ns = self.ptp_clock.get_hardware_nanoseconds()

            # 1. DMA Zero-Copy Ingestion
            incoming_spikes = self.dma_gate.extract_dma_spike_train(self.tensor_dim)

            # 2. Spiking Kernel Execution (via mlx_lif_ext binding in production)
            # Simulated Python fallback maintaining logic architecture:
            u_current = self.membrane_potentials - mx.right_shift(self.membrane_potentials, self.beta_decay)
            synaptic_input = mx.matmul(self.lif_weights, incoming_spikes)
            self.membrane_potentials = u_current + mx.astype(synaptic_input, mx.int32)
            
            # Fire logic
            spikes_out = mx.where(self.membrane_potentials > self.spike_threshold, mx.array(1, mx.int8),
                         mx.where(self.membrane_potentials < -self.spike_threshold, mx.array(-1, mx.int8), mx.array(0, mx.int8)))
                         
            # Reset membranes that fired
            self.membrane_potentials = mx.where(spikes_out == 1, self.membrane_potentials - self.spike_threshold, self.membrane_potentials)
            self.membrane_potentials = mx.where(spikes_out == -1, self.membrane_potentials + self.spike_threshold, self.membrane_potentials)

            # 3. Action Routing
            if mx.sum(mx.abs(spikes_out)).item() > 0:
                dummy_payload = "0x" + "FF" * 32
                self.phy_injector.inject_raw_frame(dummy_payload)
                self.ring_journal.append_state(spikes_out)

            tick_end_ns = self.ptp_clock.get_hardware_nanoseconds()
            execution_latency_ns = tick_end_ns - tick_start_ns
            
            if execution_latency_ns > 20000: # Tightened to 20 microseconds
                print(f"[KINEMATIC FAULT] Spiking sequence breached 20us limit: {execution_latency_ns}ns.")

    def shutdown(self):
        super().shutdown()
        self.dma_gate.close()
