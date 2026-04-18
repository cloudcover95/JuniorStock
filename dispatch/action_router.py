# juniorstock/dispatch/action_router.py
from juniorstock.nodes.web3_node import Web3ExecutionNode
from juniorstock.execution.macro_automata import DeterministicMacroAutomata
from juniorstock.hardware.power_governor import AutonomousPowerGovernor
from juniorstock.ledger.execution_audit import ExecutionLedger

class SDKActionDispatcher:
    """
    Central routing matrix for all validated triggers.
    Enforces power constraints and isolation gates before execution.
    """
    def __init__(self):
        self.web3 = Web3ExecutionNode()
        self.automata = DeterministicMacroAutomata()
        self.power_gov = AutonomousPowerGovernor()
        self.ledger = ExecutionLedger()

    def dispatch_web3_transaction(self, contract: str, payload: str, nonce: int, current_voltage: float):
        """
        Routes payload to Web3Node. Halts if LiFePO4 cutoff is breached.
        """
        if self.power_gov.calculate_compute_throttle(current_voltage) == 0:
            self.ledger.record_action("WEB3_TX", contract, {"nonce": nonce}, "HALT_VOLTAGE_LOW")
            print("[DISPATCH GATE] Action halted. Sovereign power envelope breached.")
            return False

        try:
            signed_tx = self.web3.build_and_sign_transaction(contract, payload, nonce)
            self.ledger.record_action("WEB3_TX", contract, {"nonce": nonce}, "SUCCESS")
            return signed_tx
        except Exception as e:
            self.ledger.record_action("WEB3_TX", contract, {"nonce": nonce}, f"FAIL: {e}")
            raise

    def dispatch_macro_trigger(self, hex_payload: str, current_wattage: float):
        """
        Routes directly to ATmega32u4 hardware passthrough.
        Halts if thermal bounds (>45W) are breached.
        """
        if self.power_gov.enforce_thermal_limit(current_wattage):
            self.ledger.record_action("MACRO_EXEC", "ATmega32u4", {"payload": hex_payload}, "HALT_THERMAL_LIMIT")
            print("[DISPATCH GATE] Action halted. Thermal envelope (>45W) exceeded.")
            return False

        try:
            self.automata.execute_firmware_trigger(hex_payload)
            self.ledger.record_action("MACRO_EXEC", "ATmega32u4", {"payload": hex_payload}, "SUCCESS")
            return True
        except Exception as e:
            self.ledger.record_action("MACRO_EXEC", "ATmega32u4", {"payload": hex_payload}, f"FAIL: {e}")
            raise
