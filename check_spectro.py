import numpy as np, soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
from spectro import image_to_wav

img = Image.open("assets/secret.png").convert("L").resize((256, 64))
img = (np.asarray(img) > 128).astype(np.uint8)

image_to_wav(img, "out_test.wav")
x, fs = sf.read("out_test.wav")
plt.specgram(x, NFFT=2048, Fs=fs, noverlap=1024)
plt.ylim(1500, 10500)
plt.show()