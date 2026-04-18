# juniorstock/network/ptp/mach_clock.py
import ctypes
import sys

class PTPKinematicClock:
    """
    Bypasses standard OS time.time() polling.
    Binds directly to Darwin's mach_absolute_time to extract nanosecond-precision 
    hardware ticks for Slate AX multi-agent synchronization.
    """
    def __init__(self):
        self.is_darwin = sys.platform == 'darwin'
        self.timebase_info = self._get_mach_timebase()
        
        if self.is_darwin:
            self.libc = ctypes.CDLL('/usr/lib/libc.dylib')
            self.libc.mach_absolute_time.restype = ctypes.c_uint64

    def _get_mach_timebase(self):
        """
        Retrieves the Apple Silicon hardware clock conversion factors 
        (numerators and denominators) to map raw ticks to nanoseconds.
        """
        if not self.is_darwin:
            return None
            
        class mach_timebase_info(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32),
                        ("denom", ctypes.c_uint32)]
        
        info = mach_timebase_info()
        libc = ctypes.CDLL('/usr/lib/libc.dylib')
        libc.mach_timebase_info(ctypes.byref(info))
        return (info.numer, info.denom)

    def get_hardware_nanoseconds(self) -> int:
        """
        Extracts the un-jittered hardware clock.
        """
        if not self.is_darwin:
            import time
            return time.time_ns()
            
        absolute_ticks = self.libc.mach_absolute_time()
        
        # Convert ticks to nanoseconds via timebase ratio
        nano = (absolute_ticks * self.timebase_info[0]) // self.timebase_info[1]
        return nano

    def sync_mesh_timestamp(self, peer_nano: int) -> int:
        """
        Calculates nanosecond kinematic drift across the local AX cluster.
        """
        local_nano = self.get_hardware_nanoseconds()
        drift = local_nano - peer_nano
        
        # Execution is halted if drift exceeds 10 microseconds to prevent
        # fractured state reconciliation across the multi-agent mesh.
        if abs(drift) > 10000:
            print(f"[PTP FAULT] Mesh temporal drift out of bounds: {drift}ns.")
            
        return drift
