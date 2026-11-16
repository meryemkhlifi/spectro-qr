import numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectro import image_to_wav, wav_to_image

rng = np.random.default_rng(0)
img = rng.integers(0, 2, (64, 256), dtype=np.uint8)   # ressemble à l'image chiffrée
sigmas = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
res = {}
for rep in (1, 3):
    image_to_wav(img, "ber_tmp.wav", repeat=rep)
    x, fs = sf.read("ber_tmp.wav")
    bers = []
    for s in sigmas:
        trials = []
        for seed in range(3):
            r = np.random.default_rng(100 + seed)
            sf.write("ber_noisy.wav", x + r.normal(0, s, x.shape), fs, subtype="PCM_16")
            trials.append((wav_to_image("ber_noisy.wav") != img).mean())
        bers.append(float(np.mean(trials)))
    res[rep] = bers

print("sigma  | BER x1   | BER x3")
for i, s in enumerate(sigmas):
    print(f"{s:<6} | {res[1][i]:.4f}   | {res[3][i]:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(sigmas, res[1], "o-", label="sans redondance")
plt.plot(sigmas, res[3], "s-", label="redondance x3")
plt.axhline(0.01, color="gray", ls="--", lw=0.8)
plt.xlabel("écart-type du bruit gaussien (signal pleine échelle = 1)")
plt.ylabel("BER"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("ber.png", dpi=150)