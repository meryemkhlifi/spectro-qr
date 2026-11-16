# crypto.py
import numpy as np
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

SHAPE = (64, 256)

def xor_encrypt(img_bin, K: bytes):
    """Same function encrypts and decrypts."""
    img_bin = np.asarray(img_bin, dtype=np.uint8)
    assert img_bin.shape == SHAPE, f"expected {SHAPE}, got {img_bin.shape}"
    key = HKDF(algorithm=hashes.SHA256(), length=32,
               salt=None, info=b"spectro-qr-keystream").derive(K)
    nonce = b"\x00" * 16
    enc = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor()
    n_bytes = SHAPE[0] * SHAPE[1] // 8
    ks = enc.update(b"\x00" * n_bytes)                  # raw keystream
    ks_bits = np.unpackbits(np.frombuffer(ks, dtype=np.uint8)).reshape(SHAPE)
    return (img_bin ^ ks_bits).astype(np.uint8)