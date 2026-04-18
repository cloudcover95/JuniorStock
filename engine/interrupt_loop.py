# juniorstock/engine/interrupt_loop.py
import signal
import mlx.core as mx
from juniorstock.engine.lut_loop import LUTExecutionEngine
from juniorstock.hardware.interrupts.gpio_wake import HardwareInterruptGateway
from juniorstock.math.lie_group.tangent_router import DiscreteTangentProjector

class InterruptExecutionEngine(LUTExecutionEngine):
    """
    Phase 25 Override: Dismantles the asyncio while loop.
    Thread executes `signal.pause()`, yielding entirely to the OS idle scheduler 
    until the ATmega32u4 hardware interrupt wakes it for instant execution.
    """
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.interrupt_gate = HardwareInterruptGateway()
        self.tangent_projector = DiscreteTangentProjector(hyper_dim=10) # Aligned to L2 mock tensor

    def execution_cycle_blocking(self):
        self.qos_router.pin_to_p_cores()
        print("[MACH ROUTE] Interrupt Engine pinned to M4 P-Cores. Asyncio loop bypassed.")
        
        self.interrupt_gate.arm_hardware_interrupt()

        while self.running:
            # 1. Thermal Audit 
            current_soc_temp = self.smc_governor.read_soc_temperature()
            if current_soc_temp >= self.smc_governor.critical_temp_c:
                import time
                time.sleep(5.0)
                continue

            # 2. Hardware Sleep State (0W Idle)
            # The thread halts here until the ATmega32u4 pulses the DTR line.
            self.interrupt_gate.interrupt_triggered = False
            signal.pause() 
            
            if not self.interrupt_gate.interrupt_triggered:
                continue

            # 3. WAKEUP ROUTINE (Nanosecond execution)
            tick_start_ns = self.ptp_clock.get_hardware_nanoseconds()

            # Read raw injection directly from the hardware buffer
            # (Simulated for architecture logic)
            delta_manifold = mx.random.randint(-1, 2, (1, 10), dtype=mx.int8)

            # 4. Tangent Arbitrage Projection (L1 Aligned)
            arb_signal = self.tangent_projector.calculate_tangent_arbitrage(delta_manifold)

            if mx.sum(mx.abs(arb_signal)).item() > 0:
                # Execution convergence triggered. 
                # Bypass LUT completely, execute via HDAM + PHY
                dummy_payload = "0x" + "FF" * 32
                self.phy_injector.inject_raw_frame(dummy_payload)
                
                self.ring_journal.append_state(arb_signal)

            tick_end_ns = self.ptp_clock.get_hardware_nanoseconds()
            execution_latency_ns = tick_end_ns - tick_start_ns
            
            if execution_latency_ns > 50000: # Tightened to 50 microseconds
                print(f"[KINEMATIC FAULT] Interrupt sequence breached 50us limit: {execution_latency_ns}ns.")

    def ignite(self):
        """
        Override ignite to block synchronously, bypassing asyncio overhead.
        """
        print("[ENGINE START] Initializing Hardware Interrupt routing.")
        self.running = True
        try:
            self.execution_cycle_blocking()
        except KeyboardInterrupt:
            self.running = False
            self.interrupt_gate.disarm()
            print("\n[ENGINE STOP] Local interrupt received. Hardware unarmed.")
