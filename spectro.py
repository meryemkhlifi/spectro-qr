import numpy as np
import soundfile as sf

ROWS, COLS = 64, 256
FS = 44100                      # Hz
COL_DUR = 0.05                  # 50 ms par colonne
N = int(FS * COL_DUR)           # 2205 échantillons par colonne
F_MIN, F_MAX = 2000.0, 10000.0
FREQS = np.linspace(F_MAX, F_MIN, ROWS)   # ligne 0 = fréquence la plus haute
# Phase fixe et différente pour chaque ligne : évite que les 64 sinus
# s'additionnent en phase (pic énorme, donc chaque sinus devient très faible).
PHASES = np.random.default_rng(12345).uniform(0, 2 * np.pi, ROWS)


def _load_cover(cover_wav, total):
    c, fs = sf.read(cover_wav)
    if c.ndim > 1:
        c = c.mean(axis=1)
    assert fs == FS, f"le fond sonore doit être à {FS} Hz (il est à {fs})"
    if len(c) < total:
        c = np.tile(c, int(np.ceil(total / len(c))))
    c = c[:total]
    peak = np.max(np.abs(c))
    return c / peak if peak > 0 else c


def image_to_wav(img_bin, out_wav, cover_wav=None, repeat=1,
                 pattern_gain=0.5, cover_gain=0.4):
    img_bin = np.asarray(img_bin, dtype=np.uint8)
    assert img_bin.shape == (ROWS, COLS), f"forme attendue {(ROWS, COLS)}"
    img_rep = np.tile(img_bin, (1, repeat))          # motif répété `repeat` fois
    total = COLS * repeat * N
    t = np.arange(total) / FS
    ramp = np.hanning(int(0.005 * FS) * 2 + 1)       # lissage ~5 ms contre les clics
    ramp /= ramp.sum()
    signal = np.zeros(total)
    for r in range(ROWS):
        env = np.repeat(img_rep[r].astype(float), N)
        env = np.convolve(env, ramp, mode="same")
        signal += env * np.sin(2 * np.pi * FREQS[r] * t + PHASES[r])

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    if cover_wav is None:
        signal = 0.9 * signal
    else:
        signal = pattern_gain * signal + cover_gain * _load_cover(cover_wav, total)
    sf.write(out_wav, signal, FS, subtype="PCM_16")


def wav_to_image(wav_path):
    x, fs = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == FS, f"fréquence d'échantillonnage inattendue : {fs}"
    repeat = max(1, int(round(len(x) / (COLS * N))))  # déduit de la durée du fichier
    ncols = COLS * repeat
    x = x[: ncols * N]
    if len(x) < ncols * N:
        x = np.pad(x, (0, ncols * N - len(x)))

    win = np.hanning(N)
    bins = np.round(FREQS * N / FS).astype(int)
    energy = np.zeros((ROWS, ncols))
    for c in range(ncols):
        seg = x[c * N:(c + 1) * N] * win
        spec = np.abs(np.fft.rfft(seg))
        energy[:, c] = np.max([spec[bins - 1], spec[bins], spec[bins + 1]], axis=0)

    energy = energy.reshape(ROWS, repeat, COLS).mean(axis=1)  # moyenne des répétitions
    thr = 0.5 * np.percentile(energy, 99)   # seuil : moitié d'un pixel allumé
    return (energy > thr).astype(np.uint8)