from PIL import Image, ImageDraw, ImageFont

W, H = 256, 64
img = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=44)
text = "HI"
box = d.textbbox((0, 0), text, font=font)
x = (W - (box[2] - box[0])) // 2 - box[0]
y = (H - (box[3] - box[1])) // 2 - box[1]
d.text((x, y), text, fill=255, font=font)
img.save("assets/secret.png")