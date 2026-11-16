import numpy as np
import soundfile as sf

ROWS, COLS = 64, 256
FS = 44100                      # Hz
COL_DUR = 0.05                  # 50 ms par colonne
N = int(FS * COL_DUR)           # 2205 échantillons par colonne
F_MIN, F_MAX = 2000.0, 10000.0
FREQS = np.linspace(F_MAX, F_MIN, ROWS)   # ligne 0 = fréquence la plus haute


def image_to_wav(img_bin, out_wav, cover_wav=None):
    img_bin = np.asarray(img_bin, dtype=np.uint8)
    assert img_bin.shape == (ROWS, COLS), f"forme attendue {(ROWS, COLS)}"
    if cover_wav is not None:
        raise NotImplementedError("fond sonore : jour 2")

    total = COLS * N
    t = np.arange(total) / FS
    ramp = np.hanning(int(0.005 * FS) * 2 + 1)   # lissage ~5 ms contre les clics
    ramp /= ramp.sum()
    signal = np.zeros(total)
    for r in range(ROWS):
        env = np.repeat(img_bin[r].astype(float), N)
        env = np.convolve(env, ramp, mode="same")
        signal += env * np.sin(2 * np.pi * FREQS[r] * t)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.9 * signal / peak
    sf.write(out_wav, signal, FS, subtype="PCM_16")
def wav_to_image(wav_path):
    x, fs = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == FS, f"fréquence d'échantillonnage inattendue : {fs}"
    x = x[: COLS * N]
    if len(x) < COLS * N:
        x = np.pad(x, (0, COLS * N - len(x)))

    win = np.hanning(N)
    bins = np.round(FREQS * N / FS).astype(int)
    energy = np.zeros((ROWS, COLS))
    for c in range(COLS):
        seg = x[c * N:(c + 1) * N] * win
        spec = np.abs(np.fft.rfft(seg))
        energy[:, c] = np.max([spec[bins - 1], spec[bins], spec[bins + 1]], axis=0)

    thr = 0.5 * energy.max()
    return (energy > thr).astype(np.uint8)