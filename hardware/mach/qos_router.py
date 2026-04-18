# juniorstock/hardware/mach/qos_router.py
import ctypes
import sys

class AppleSiliconQoSRouter:
    """
    Direct bindings to Darwin libSystem.dylib.
    Forces the macOS kernel to pin the current thread to specific silicon cores
    (P-Cores for latency-critical math, E-Cores for background I/O).
    """
    # macOS QoS Classes (sys/qos.h)
    QOS_CLASS_USER_INTERACTIVE = 0x21  # Max priority -> P-Cores
    QOS_CLASS_USER_INITIATED = 0x19
    QOS_CLASS_DEFAULT = 0x15
    QOS_CLASS_UTILITY = 0x11
    QOS_CLASS_BACKGROUND = 0x09        # Min priority -> E-Cores

    def __init__(self):
        if sys.platform != 'darwin':
            self.lib = None
            return
            
        try:
            self.lib = ctypes.CDLL('/usr/lib/libSystem.dylib')
            self.lib.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
            self.lib.pthread_set_qos_class_self_np.restype = ctypes.c_int
        except Exception as e:
            print(f"[MACH FAULT] Failed to load Darwin bindings: {e}")
            self.lib = None

    def pin_to_p_cores(self):
        """
        Forces thread execution onto M4/M1 Performance Cores.
        Critical for FSD logic gates and macro-automata triggers.
        """
        if self.lib:
            result = self.lib.pthread_set_qos_class_self_np(self.QOS_CLASS_USER_INTERACTIVE, 0)
            if result != 0:
                print(f"[MACH FAULT] P-Core pinning failed. Errno: {result}")

    def pin_to_e_cores(self):
        """
        Banishes thread to M4/M1 Efficiency Cores.
        Critical for Parquet flushing and AX Mesh UDP broadcasting to preserve thermal mass.
        """
        if self.lib:
            result = self.lib.pthread_set_qos_class_self_np(self.QOS_CLASS_BACKGROUND, 0)
            if result != 0:
                print(f"[MACH FAULT] E-Core pinning failed. Errno: {result}")
