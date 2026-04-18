# juniorstock/licensing/hwid_lock.py
import subprocess
import hashlib
import sys

class HardwareFingerprint:
    """
    Extracts the immutable IOPlatformUUID directly from the Apple Silicon SoC.
    Hashes the output to prevent raw UUID leakage in the enterprise payload.
    """
    @staticmethod
    def generate_soc_hash() -> str:
        if sys.platform != 'darwin':
            # Fallback for non-macOS local testing, though strictly violates deployment target
            return hashlib.sha256(b"NON_APPLE_SILICON_FALLBACK").hexdigest()

        try:
            # Bypass Python standard libraries for direct IOKit registry extraction
            cmd = "ioreg -rd1 -c IOPlatformExpertDevice | awk '/IOPlatformUUID/ { split($0, line, \"\\\"\"); printf(\"%s\\n\", line[4]); }'"
            uuid_raw = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
            
            if not uuid_raw:
                raise ValueError("IOPlatformUUID extraction failed.")

            # Append static salt to prevent rainbow table attacks on the HWID
            salted_uuid = uuid_raw + "_JUNIORCLOUD_EDGE"
            return hashlib.sha256(salted_uuid.encode('utf-8')).hexdigest()
            
        except Exception as e:
            print(f"[HWID FAULT] Unable to establish SoC fingerprint. ERR: {e}")
            return None
