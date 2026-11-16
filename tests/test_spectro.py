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
def _pink(n, seed=3):
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(rng.normal(size=n))
    fr = np.fft.rfftfreq(n, 1 / 44100); fr[0] = 1
    x = np.fft.irfft(F / np.sqrt(fr), n)
    return x / np.abs(x).max()


def test_cover_random(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 2, (64, 256), dtype=np.uint8)
    cover = str(tmp_path / "cover.wav")
    sf.write(cover, _pink(44100 * 14), 44100, subtype="PCM_16")
    p = str(tmp_path / "c.wav")
    image_to_wav(img, p, cover_wav=cover)
    assert (wav_to_image(p) == img).mean() > 0.99


def test_repeat_helps(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 2, (64, 256), dtype=np.uint8)
    errs = []
    for rep in (1, 3):
        p = str(tmp_path / f"r{rep}.wav")
        image_to_wav(img, p, repeat=rep)
        x, fs = sf.read(p)
        r = np.random.default_rng(5)
        sf.write(p, x + r.normal(0, 0.2, x.shape), fs, subtype="PCM_16")
        errs.append((wav_to_image(p) != img).mean())
    assert errs[1] < errs[0] / 3