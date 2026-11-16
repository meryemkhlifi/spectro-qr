import sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

src, dst = sys.argv[1], sys.argv[2]
x, fs = sf.read(src)
if x.ndim > 1:
    x = x.mean(axis=1)
g = gcd(44100, int(fs))
x = resample_poly(x, 44100 // g, int(fs) // g)
x = x[: 44100 * 40]                       # 40 s max
x = 0.9 * x / np.max(np.abs(x))
sf.write(dst, x, 44100, subtype="PCM_16")
print("ok :", dst, f"{len(x)/44100:.1f} s")