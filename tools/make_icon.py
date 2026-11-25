"""Draw the app icon (app/assets/icon.png and icon.ico): a frosted pane on a dusk-blue tile,
a Japanese あ turning into an English A. Also the check mark of the panel's checkboxes
(app/assets/check.png)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "app" / "assets"


def font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(size: int = 512) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    # the tile: a vertical gradient, round corners
    grad = Image.new("RGBA", (s, s))
    top, bottom = (52, 92, 168), (24, 38, 84)
    px = grad.load()
    for y in range(s):
        t = y / (s - 1)
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(s):
            px[x, y] = (*c, 255)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.2), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # the frosted pane, a light border
    pane = (int(s * 0.12), int(s * 0.56), int(s * 0.88), int(s * 0.86))
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle(pane, radius=int(s * 0.06), fill=(255, 255, 255, 70))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(s * 0.01)))
    d.rounded_rectangle(
        pane, radius=int(s * 0.06), outline=(255, 255, 255, 150), width=max(2, s // 128)
    )
    # あ above, A on the pane
    jp = font(["YuGothB.ttc", "meiryob.ttc", "msgothic.ttc"], int(s * 0.34))
    en = font(["segoeuib.ttf", "arialbd.ttf"], int(s * 0.25))
    d.text((s * 0.5, s * 0.30), "あ", font=jp, fill=(255, 255, 255, 235), anchor="mm")
    d.text((s * 0.5, s * 0.712), "A", font=en, fill=(255, 236, 170, 255), anchor="mm")
    return img


def check(size: int = 28) -> Image.Image:
    """A white check mark, drawn 4x and scaled down (smooth edges)."""
    k = 4
    img = Image.new("RGBA", (size * k, size * k), (0, 0, 0, 0))
    pts = [(0.22, 0.52), (0.42, 0.72), (0.78, 0.30)]
    ImageDraw.Draw(img).line(
        [(x * size * k, y * size * k) for x, y in pts], fill=(255, 255, 255, 255),
        width=int(size * k * 0.13), joint="curve",
    )  # fmt: skip
    return img.resize((size, size), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    check().save(OUT / "check.png")
    big = draw(512)
    big.save(OUT / "icon.png")
    big.save(
        OUT / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("written", OUT)
