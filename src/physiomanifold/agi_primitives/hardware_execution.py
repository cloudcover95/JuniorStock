# src/physiomanifold/agi_primitives/hardware_execution.py

import serial
import logging
import mlx.core as mx

class ATmegaHardwareLock:
    """
    Zero-latency deterministic bridge to physical execution.
    Bypasses OS TCP/IP stack for direct serial payload injection.
    """
    def __init__(self, port: str = "/dev/cu.usbmodem1101", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None

    def engage_lock(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.01)
            logging.info(f"Hardware Lock Engaged on {self.port}")
        except serial.SerialException:
            logging.warning(f"Hardware Lock Offline. Simulating execution on {self.port}")

    def execute_trade_signal(self, koopman_matrix: mx.array, free_energy_state: float):
        """
        Translates mathematical convergence into physical byte code.
        """
        # Extract leading eigenvalue proxy (trace of the reduced matrix)
        momentum_proxy = mx.sum(mx.diag(koopman_matrix)).item()
        
        # Signal Generation Logic
        if free_energy_state < 0.1 and momentum_proxy > 1.05:
            payload = b'\x01' # Deterministic BUY trigger
            signal_type = "LONG"
        elif free_energy_state < 0.1 and momentum_proxy < 0.95:
            payload = b'\x02' # Deterministic SELL trigger
            signal_type = "SHORT"
        else:
            payload = b'\x00' # HOLD trigger
            signal_type = "HOLD"

        # Hardware execution
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(payload)
            logging.info(f"[PHYSICAL LAYER] Transmitted {signal_type} instruction via serial.")
        else:
            logging.info(f"[SIMULATION] Required {signal_type} instruction. HW offline.")
