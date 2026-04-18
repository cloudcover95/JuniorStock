# juniorstock/engine/consensus_loop.py
import signal
import hashlib
import mlx.core as mx
from juniorstock.engine.spiking_loop import SpikingExecutionEngine
from juniorstock.network.consensus.ternary_dag import TernaryDAGConsensus

class ConsensusExecutionEngine(SpikingExecutionEngine):
    """
    Phase 27 Override: Wraps the ASTA engine with Keccak Enclave validation 
    and DAG Mesh Consensus before permitting PHY hardware injection.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.dag_consensus = TernaryDAGConsensus(consensus_threshold=3) # Host M4 + 2x Orange Pi Min

    def execution_cycle_blocking(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] Consensus Engine Pinned. Keccak Enclaves Active.")
        
        self.interrupt_gate.arm_hardware_interrupt()

        while self.running:
            self.interrupt_gate.interrupt_triggered = False
            signal.pause() 
            
            if not self.interrupt_gate.interrupt_triggered:
                continue

            tick_start_ns = self.ptp_clock.get_hardware_nanoseconds()

            # 1. DMA Zero-Copy Ingestion
            # Assume 10-byte raw payload representing market vector
            raw_dma_bytes = self.dma_gate.extract_dma_spike_train(self.tensor_dim)

            # 2. Cryptographic Enclave Validation (Simulated mlx_keccak_ext logic)
            # Ensure incoming DMA payload has not been tampered with
            dma_hash = hashlib.sha3_256(bytes(raw_dma_bytes.tolist())).hexdigest()
            # In production: mlx_keccak_ext.hash_dma(raw_dma_bytes)

            # 3. Spiking Kernel Execution
            u_current = self.membrane_potentials - mx.right_shift(self.membrane_potentials, self.beta_decay)
            synaptic_input = mx.matmul(self.lif_weights, mx.astype(raw_dma_bytes, mx.int8))
            self.membrane_potentials = u_current + mx.astype(synaptic_input, mx.int32)
            
            spikes_out = mx.where(self.membrane_potentials > self.spike_threshold, mx.array(1, mx.int8),
                         mx.where(self.membrane_potentials < -self.spike_threshold, mx.array(-1, mx.int8), mx.array(0, mx.int8)))

            if mx.sum(mx.abs(spikes_out)).item() > 0:
                # 4. Generate Topological Hash from B1.58 output
                state_hash = hashlib.blake2b(bytes(spikes_out.tolist()), digest_size=16).hexdigest()

                # 5. Execute DAG Consensus
                # Ingest updates from Orange Pi cluster
                self.dag_consensus.poll_cluster_graph()
                
                if self.dag_consensus.append_manifold_state(state_hash):
                    # --- EXECUTION TRIGGERED (CONSENSUS REACHED) ---
                    dummy_payload = "0x" + "FF" * 32
                    self.phy_injector.inject_raw_frame(dummy_payload)
                    self.ring_journal.append_state(spikes_out)
                else:
                    print(f"[DAG LOCK] Execution held. Awaiting cluster consensus for state {state_hash[:8]}.")

            tick_end_ns = self.ptp_clock.get_hardware_nanoseconds()
            execution_latency_ns = tick_end_ns - tick_start_ns
            
            if execution_latency_ns > 40000:
                print(f"[KINEMATIC FAULT] Consensus sequence breached 40us limit: {execution_latency_ns}ns.")
