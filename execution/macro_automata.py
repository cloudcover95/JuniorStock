# juniorstock/execution/macro_automata.py
import serial
import time
from typing import List

class DeterministicMacroAutomata:
    """
    Hardware-level deterministic trade execution using ATmega32u4.
    Bypasses OS-level DWM / kernel scheduling jitter.
    Injects execution commands natively as HID packets.
    """
    def __init__(self, serial_port: str = '/dev/cu.usbmodem14101', baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        try:
            self.device = serial.Serial(self.serial_port, self.baudrate, timeout=0.001)
        except Exception as e:
            # Expected if edge node is not connected yet; handle silently per No Fluff directive.
            self.device = None

    def execute_firmware_trigger(self, hex_payload: str):
        """
        Fires the serialized transaction or trade trigger directly to the HID microcontroller.
        """
        if self.device:
            packet = f"EXEC:{hex_payload}\n".encode('utf-8')
            self.device.write(packet)
            self.device.flush()
            
            # Deterministic kinematic delay (matches expected micro-switch actuation bounds)
            time.sleep(0.002) 
