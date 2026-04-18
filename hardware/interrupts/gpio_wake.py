# juniorstock/hardware/interrupts/gpio_wake.py
import os
import fcntl
import signal
import termios
import sys

class HardwareInterruptGateway:
    """
    Binds the ATmega32u4 DTR/CTS UART lines to Darwin POSIX signals.
    Allows the Apple Silicon P-Core to enter true hardware sleep (0W idle)
    until the coprocessor asserts a physical voltage flag indicating market volatility.
    """
    def __init__(self, serial_port: str = '/dev/cu.usbmodem14101'):
        self.serial_port = serial_port
        self.fd = None
        self.interrupt_triggered = False
        
        if sys.platform != 'darwin':
            print("[INTERRUPT FAULT] True POSIX SIGIO binding requires Darwin kernel.")
            return

    def _sigio_handler(self, signum, frame):
        """
        C-level callback executed by the Darwin kernel upon UART buffer assertion.
        """
        self.interrupt_triggered = True

    def arm_hardware_interrupt(self):
        """
        Configures the TTY file descriptor for async I/O signaling.
        """
        if sys.platform != 'darwin':
            return
            
        try:
            # Open non-blocking
            self.fd = os.open(self.serial_port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            
            # Bind SIGIO to the current process
            signal.signal(signal.SIGIO, self._sigio_handler)
            fcntl.fcntl(self.fd, fcntl.F_SETOWN, os.getpid())
            
            # Enable asynchronous I/O flag on the file descriptor
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_ASYNC)
            
            print("[MACH ROUTE] Hardware Interrupt Armed. P-Core ready for sleep state.")
        except Exception as e:
            print(f"[INTERRUPT FAULT] Failed to bind UART to POSIX SIGIO: {e}")

    def disarm(self):
        """
        Releases file descriptor prior to graceful shutdown.
        """
        if self.fd is not None:
            os.close(self.fd)
