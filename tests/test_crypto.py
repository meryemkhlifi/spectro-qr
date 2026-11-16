import os, numpy as np
from crypto import xor_encrypt

def _img():
    return np.random.randint(0, 2, (64, 256), dtype=np.uint8)

def test_roundtrip():
    K = os.urandom(32); m = _img()
    assert np.array_equal(xor_encrypt(xor_encrypt(m, K), K), m)

def test_wrong_key_gives_noise():
    m = _img(); c = xor_encrypt(m, os.urandom(32))
    wrong = xor_encrypt(c, os.urandom(32))
    assert 0.4 < (wrong != m).mean() < 0.6      # ~50% bit errors = noise