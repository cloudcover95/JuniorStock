# juniorstock/hardware/power_governor.py
import mlx.core as mx

class AutonomousPowerGovernor:
    """
    Hardware-level telemetry monitor targeting 48V LiFePO4 off-grid environments.
    Dynamically throttles the ATML b1.58 tensor processing batches based on 
    available localized power capacity and transient voltage dips.
    """
    def __init__(self, v_nominal: float = 51.2, v_cutoff: float = 48.0):
        self.v_nominal = v_nominal
        self.v_cutoff = v_cutoff
        self.base_batch_size = 1000

    def calculate_compute_throttle(self, current_voltage: float) -> int:
        """
        Scales the processing batch size logarithmically as voltage approaches the 
        LiFePO4 knee/cutoff curve to preserve sovereign node uptime.
        """
        if current_voltage <= self.v_cutoff:
            return 0  # Absolute compute halt. Reserve power for memory persistence.
            
        # Hardware capacity ratio
        capacity_ratio = (current_voltage - self.v_cutoff) / (self.v_nominal - self.v_cutoff)
        
        # Apply scaling via MLX array for strict type consistency with engine config
        throttle_factor = mx.array([capacity_ratio])
        
        # Determine throttled batch size via MLX floor
        adjusted_batch = mx.floor(mx.maximum(mx.array([100.0]), self.base_batch_size * throttle_factor))
        
        return int(adjusted_batch.item())

    def enforce_thermal_limit(self, current_wattage: float, power_limit: float = 45.0) -> bool:
        """
        Hard logic gate to ensure M1/M4 SoC remains within the specified 45W envelope.
        Returns True if threshold is breached, triggering macro-automata killswitch.
        """
        return current_wattage > power_limit
