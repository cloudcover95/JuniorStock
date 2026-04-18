# juniorstock/crypto/state_encryption.py
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from juniorstock.security.path_isolation import EnvironmentIsolationGate

class SovereignStateCipher:
    """
    AES-GCM encryption for at-rest topological memory.
    Ensures that physical compromise of the edge node does not leak 
    the proprietary b1.58 historical Betti signatures.
    """
    def __init__(self, key_path: str = "juniorstock/crypto/.cipher_key"):
        EnvironmentIsolationGate.verify_io_path(key_path)
        self.key_path = key_path
        self._key = self._load_or_generate_key()
        self.aesgcm = AESGCM(self._key)

    def _load_or_generate_key(self) -> bytes:
        """
        Retrieves or generates the 256-bit AES master key.
        """
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        else:
            key = AESGCM.generate_key(bit_length=256)
            with open(self.key_path, "wb") as f:
                f.write(key)
            # Enforce strict file permissions for the key
            os.chmod(self.key_path, 0o600)
            return key

    def encrypt_manifold_buffer(self, serialized_buffer: bytes) -> bytes:
        """
        Encrypts the Parquet buffer prior to physical disk flush.
        Requires a unique 96-bit nonce per operation.
        """
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, serialized_buffer, None)
        return nonce + ciphertext

    def decrypt_manifold_buffer(self, encrypted_payload: bytes) -> bytes:
        """
        Decrypts payload during historical replay ingestion.
        """
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)
