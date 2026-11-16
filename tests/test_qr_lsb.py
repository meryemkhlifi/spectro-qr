import os, numpy as np, pytest
from PIL import Image
from qr_lsb import embed_qr, extract_key

@pytest.fixture
def cover(tmp_path):
    p = tmp_path / "cover.png"
    arr = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    return str(p)

def test_roundtrip(cover, tmp_path):
    K = os.urandom(32); out = str(tmp_path / "s.png")
    embed_qr(cover, K, "pwd", out)
    assert extract_key(out, "pwd") == K

def test_wrong_password(cover, tmp_path):
    out = str(tmp_path / "s.png")
    embed_qr(cover, os.urandom(32), "pwd", out)
    with pytest.raises(ValueError):
        extract_key(out, "bad")

def test_max_diff_is_1(cover, tmp_path):
    out = str(tmp_path / "s.png")
    embed_qr(cover, os.urandom(32), "pwd", out)
    a = np.array(Image.open(cover)).astype(int)
    b = np.array(Image.open(out)).astype(int)
    assert np.abs(a - b).max() <= 1
