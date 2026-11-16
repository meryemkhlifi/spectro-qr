# decode.py
import sys, argparse, getpass
import numpy as np
from PIL import Image
from crypto import xor_encrypt
from qr_lsb import extract_key
from spectro import wav_to_image

ap = argparse.ArgumentParser()
ap.add_argument("--png", default="meme_stego.png")
ap.add_argument("--wav", default="out.wav")
ap.add_argument("--out", default="secret_decoded.png")
a = ap.parse_args()
password = getpass.getpass("Password: ")

try:
    K = extract_key(a.png, password)                         # 1. key from the PNG
except ValueError as e:
    sys.exit(f"FAILED: {e}")

C = wav_to_image(a.wav)                                      # 2. encrypted image from the WAV
M = xor_encrypt(C, K)                                        # 3. decrypt
Image.fromarray(((1 - M) * 255).astype(np.uint8)).save(a.out)   # 4. back to black on white
print("Saved", a.out)