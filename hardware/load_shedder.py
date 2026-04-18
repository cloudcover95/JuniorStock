# juniorstock/hardware/load_shedder.py
import os
import signal
from typing import List

class KineticLoadShedder:
    """
    Hardware-level process governor.
    Executes selective SIGTERM/SIGKILL strikes against non-critical SDK sub-agents
    when 48V LiFePO4 transient voltage drops breach critical bounds.
    """
    def __init__(self, critical_voltage: float = 48.2):
        self.critical_voltage = critical_voltage
        self.non_critical_pids: List[int] = []

    def register_agent(self, pid: int):
        """
        Registers secondary bots (e.g., UI telemetry, offline replay engines)
        as sacrificial processes during low-power events.
        """
        if pid not in self.non_critical_pids:
            self.non_critical_pids.append(pid)

    def evaluate_and_shed(self, current_voltage: float):
        """
        Evaluates the power envelope. If breached, systematically purges 
        sacrificial processes to reserve thermal mass for the core OmniMath kernel.
        """
        if current_voltage <= self.critical_voltage:
            print(f"[SHEDDER ACTIVE] Voltage {current_voltage}V below limit. Purging non-critical mesh nodes.")
            for pid in self.non_critical_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f" -> SIGTERM dispatched to PID {pid}")
                except ProcessLookupError:
                    pass
            self.non_critical_pids.clear()
