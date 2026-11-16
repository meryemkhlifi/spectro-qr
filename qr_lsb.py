import hashlib
import numpy as np
import cv2
import qrcode
import qrcode.constants
from PIL import Image

def _make_qr(K: bytes) -> np.ndarray:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,
                       border=4)
    qr.add_data(K.hex())
    qr.make(fit=True)
    m = np.array(qr.get_matrix(), dtype=np.uint8)       # 1 = module noir
    img = (1 - m).astype(np.uint8)                       # 1 = blanc, 0 = noir
    return cv2.resize(img, (QR_SIDE, QR_SIDE),
                      interpolation=cv2.INTER_NEAREST)

SALT = b"spectro-qr-v1"          # fixed salt (from the plan)
QR_SIDE = 400                    # fixed QR size in pixels, so extraction knows n_bits

def _positions(password: str, n_pixels: int, n_bits: int) -> np.ndarray:
    seed = hashlib.pbkdf2_hmac("sha256", password.encode(), SALT, 200_000)
    rng = np.random.default_rng(int.from_bytes(seed, "big"))
    return rng.permutation(n_pixels)[:n_bits]


def embed_qr(cover_png: str, K: bytes, password: str, out_png: str) -> None:
    cover = np.array(Image.open(cover_png).convert("RGB"))
    h, w, _ = cover.shape
    bits = _make_qr(K).flatten()
    assert bits.size <= h * w, "Cover too small for the QR (capacity)"
    pos = _positions(password, h * w, bits.size)
    blue = cover[:, :, 2].flatten()
    blue[pos] = (blue[pos] & 0xFE) | bits
    cover[:, :, 2] = blue.reshape(h, w)
    Image.fromarray(cover).save(out_png, format="PNG")   # PNG only, never JPEG

def extract_key(stego_png: str, password: str) -> bytes:
    stego = np.array(Image.open(stego_png).convert("RGB"))
    h, w, _ = stego.shape
    n_bits = QR_SIDE * QR_SIDE
    pos = _positions(password, h * w, n_bits)
    bits = (stego[:, :, 2].flatten()[pos] & 1).astype(np.uint8)
    qr_img = (bits.reshape(QR_SIDE, QR_SIDE) * 255).astype(np.uint8)
    qr_img = cv2.copyMakeBorder(qr_img, 40, 40, 40, 40,
                                cv2.BORDER_CONSTANT, value=255)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_img)
    if not text:
        raise ValueError("Extraction failed: wrong password or no QR in image")
    try:
        K = bytes.fromhex(text)
    except ValueError:
        raise ValueError("Extraction failed: wrong password (invalid data)")
    if len(K) != 32:
        raise ValueError("Extraction failed: invalid key length")
    return K
