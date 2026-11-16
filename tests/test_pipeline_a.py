import os, numpy as np
from PIL import Image
from crypto import xor_encrypt
from qr_lsb import embed_qr, extract_key

def test_full_A_side(tmp_path):
    cover = tmp_path / "c.png"
    Image.fromarray(np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)).save(cover)
    secret = np.random.randint(0, 2, (64, 256), dtype=np.uint8)
    K = os.urandom(32)

    C = xor_encrypt(secret, K)                    # what B receives
    embed_qr(str(cover), K, "pwd", str(tmp_path / "s.png"))

    K2 = extract_key(str(tmp_path / "s.png"), "pwd")
    assert np.array_equal(xor_encrypt(C, K2), secret)