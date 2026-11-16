import numpy as np
import soundfile as sf
from PIL import Image
from spectro import image_to_wav, wav_to_image


def load_secret():
    img = Image.open("assets/secret.png").convert("L").resize((256, 64))
    return (np.asarray(img) > 128).astype(np.uint8)


def test_roundtrip_secret(tmp_path):
    img = load_secret()
    p = str(tmp_path / "t.wav")
    image_to_wav(img, p)
    assert (wav_to_image(p) == img).all()


def test_roundtrip_random(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 2, (64, 256), dtype=np.uint8)
    p = str(tmp_path / "r.wav")
    image_to_wav(img, p)
    assert (wav_to_image(p) == img).mean() > 0.999


def test_noise_robustness(tmp_path):
    img = load_secret()
    p = str(tmp_path / "n.wav")
    image_to_wav(img, p)
    x, fs = sf.read(p)
    rng = np.random.default_rng(1)
    sf.write(p, x + rng.normal(0, 0.02, x.shape), fs, subtype="PCM_16")
    assert (wav_to_image(p) == img).mean() > 0.99