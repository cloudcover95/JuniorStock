# juniorstock/licensing/drm_verifier.py
import json
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
from juniorstock.licensing.hwid_lock import HardwareFingerprint

class EnterpriseLicenseGate:
    """
    Offline cryptographic verifier. 
    Requires an Ed25519 public key injected during enterprise provisioning.
    Validates that the license payload matches the physical M1/M4 SoC.
    """
    def __init__(self, public_key_hex: str = None):
        # Default JuniorCloud LLC Public Key (Placeholder for compilation)
        self.pub_key_hex = public_key_hex or "b7a3c12...[REDACTED_FOR_COMPILATION]"
        self.cached_state = False

    def _load_public_key(self) -> ed25519.Ed25519PublicKey:
        try:
            raw_bytes = bytes.fromhex(self.pub_key_hex)
            return ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)
        except Exception as e:
            raise ValueError(f"[DRM FAULT] Corrupt enterprise public key. ERR: {e}")

    def verify_license_payload(self, license_jwt: str) -> bool:
        """
        Parses base64 payload: { "hwid": "...", "tier": "ENTERPRISE", "exp": 179... }
        Verifies signature against local HWID.
        """
        try:
            payload_b64, signature_b64 = license_jwt.split('.')
            payload_json = base64.b64decode(payload_b64).decode('utf-8')
            payload_data = json.loads(payload_json)
            
            signature = base64.b64decode(signature_b64)
            pub_key = self._load_public_key()
            
            # 1. Cryptographic validation
            pub_key.verify(signature, payload_b64.encode('utf-8'))
            
            # 2. SoC Fingerprint validation
            local_hwid = HardwareFingerprint.generate_soc_hash()
            if payload_data.get("hwid") != local_hwid:
                print(f"[DRM FAULT] License HWID mismatch. SoC spoofing detected.")
                self.cached_state = False
                return False

            # 3. Check expiration (if applicable for time-bound leases)
            # Logic omitted for brevity; assume perpetual edge license for standard deploy.

            print(f"[DRM SUCCESS] Enterprise license verified. Tier: {payload_data.get('tier')}")
            self.cached_state = True
            return True

        except InvalidSignature:
            print("[DRM FAULT] Invalid cryptographic signature. Unauthorized payload.")
        except Exception as e:
            print(f"[DRM FAULT] Payload malformed. ERR: {e}")
            
        self.cached_state = False
        return False
