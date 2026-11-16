# encode.py
import os, argparse, getpass
import numpy as np
from PIL import Image
from crypto import xor_encrypt
from qr_lsb import embed_qr
from spectro import image_to_wav

ap = argparse.ArgumentParser()
ap.add_argument("--meme", default="assets/meme.png")
ap.add_argument("--secret", default="assets/secret.png")
ap.add_argument("--cover-wav", default=None)
ap.add_argument("--out-wav", default="out.wav")
ap.add_argument("--out-png", default="meme_stego.png")
a = ap.parse_args()
password = getpass.getpass("Password: ")

K = os.urandom(32)                                           # 1. fresh key
g = Image.open(a.secret).convert("L").resize((256, 64))
M = (np.array(g) < 128).astype(np.uint8)                     # 2. dark pixel = 1, shape (64, 256)
C = xor_encrypt(M, K)                                        # 3. encrypt
image_to_wav(C, a.out_wav, cover_wav=a.cover_wav)            # 4. layer 2 (B)
embed_qr(a.meme, K, password, a.out_png)                     # 5. layer 1 (A)
print("OK ->", a.out_wav, a.out_png)