# juniorstock/execution/phy/phy_injector.py
import os
import termios
import fcntl
import time

class PHYNetworkInjector:
    """
    Direct Darwin syscall implementation of the PHY Injector.
    Bypasses pyserial bloat. Utilizes native termios for 2Mbaud UART.
    """
    def __init__(self, serial_port: str = '/dev/cu.usbmodem14101', baudrate: int = 2000000):
        self.serial_port = serial_port
        self.baudrate = getattr(termios, f"B{baudrate}", termios.B230400)
        self.fd = None
        
        try:
            self.fd = os.open(self.serial_port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            attrs = termios.tcgetattr(self.fd)
            # Set raw mode, 8N1, disable flow control
            attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP | termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON)
            attrs[1] &= ~termios.OPOST
            attrs[2] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
            attrs[3] &= ~(termios.CSIZE | termios.PARENB)
            attrs[3] |= termios.CS8
            # Set Baud
            termios.cfsetispeed(attrs, self.baudrate)
            termios.cfsetospeed(attrs, self.baudrate)
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        except Exception as e:
            # Fallback for headless/no-device state
            self.fd = None

    def inject_raw_frame(self, signed_hex_tx: str):
        if self.fd is not None:
            packet = f"RAW:{signed_hex_tx}\n".encode('ascii')
            # Flush and Write
            termios.tcflush(self.fd, termios.TCIOFLUSH)
            os.write(self.fd, packet)
            time.sleep(0.001)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
