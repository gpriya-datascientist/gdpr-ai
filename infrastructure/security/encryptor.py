"""
Layer: INFRASTRUCTURE
Imports allowed: domain + cryptography
Purpose: AES-256 encryption at rest using Fernet (symmetric).
"""
from domain.exceptions import EncryptionError
from domain.interfaces import IEncryptor


class FernetEncryptor(IEncryptor):
    """AES-256 CBC via Fernet — encrypt all documents at rest."""

    def __init__(self, key: str) -> None:
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except ImportError as e:
            raise EncryptionError("cryptography package not installed") from e
        except Exception as e:
            raise EncryptionError(f"Invalid encryption key: {e}") from e

    def encrypt(self, plaintext: str) -> str:
        try:
            return self._fernet.encrypt(plaintext.encode()).decode()
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}") from e

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet key — run once and store in .env"""
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()
