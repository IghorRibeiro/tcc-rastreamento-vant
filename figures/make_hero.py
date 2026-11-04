"""Compose the README cover (assets/hero.png), the animated preview and the GitHub social preview.

Band (a): hybrid architecture on the DUT Anti-UAV (YOLO26n detection + CSRT), the top of Figure 2 of the TCC.
Band (b): baseline x HIT-Tracker on Track 3, one frame of a side-by-side video of the same sequence.

Inputs are images and a video that are not stored in the repository (they contain frames of the public
datasets); pass their paths on the command line.

    python figures/make_hero.py --hybrid-band band_hybrid.png --mot-video side_by_side.mp4 --frame-time 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAVY = (11, 37, 69)
INK = (30, 41, 59)
MUTED = (100, 116, 139)
PAPER = (255, 255, 255)
FONT_CANDIDATES = ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
FONT_BOLD_CANDIDATES = ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Bold.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]

WIDTH = 2000
MARGIN = 28


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=1 if (bold and path.endswith(".ttc")) else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def fit_width(image: Image.Image, width: int) -> Image.Image:
    return image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)


def extract_frame(video: Path, second: float, out: Path) -> Image.Image:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(video), "-frames:v", "1", str(out)],
                   check=True)
    return Image.open(out).convert("RGB")


def build_hero(band_a: Image.Image, band_b: Image.Image, out: Path) -> Image.Image:
    inner = WIDTH - 2 * MARGIN
    a, b = fit_width(band_a.convert("RGB"), inner), fit_width(band_b.convert("RGB"), inner)
    caption_h = 64
    height = MARGIN + caption_h + a.height + 18 + caption_h + b.height + MARGIN
    canvas = Image.new("RGB", (WIDTH, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    draw.text((MARGIN, y + 8), "(a)  Alvo único, visível · DUT Anti-UAV", font=font(34, True), fill=INK)
    y += caption_h
    canvas.paste(a, (MARGIN, y))
    y += a.height + 18
    draw.text((MARGIN, y + 8), "(b)  Enxames, infravermelho · Track 3 do 4º Anti-UAV Challenge", font=font(34, True),
              fill=INK)
    y += caption_h
    canvas.paste(b, (MARGIN, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    return canvas


def build_social(band_b: Image.Image, out: Path) -> None:
    canvas = Image.new("RGB", (1280, 640), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.text((56, 44), "Rastreamento de drones usando", font=font(52, True), fill=(255, 255, 255))
    draw.text((56, 106), "visão computacional e redes neurais profundas", font=font(52, True), fill=(255, 255, 255))
    draw.text((56, 184), "Arquiteturas leves em múltiplos cenários · TCC CApA-Of 2026 · CIAA · Marinha do Brasil",
              font=font(26), fill=(186, 205, 230))
    shot = fit_width(band_b.convert("RGB"), 1280 - 112)
    shot = shot.crop((0, 0, shot.width, min(shot.height, 640 - 240 - 28)))
    canvas.paste(shot, (56, 240))
    canvas.save(out, optimize=True)


def build_gif(video: Path, out: Path, start: float, seconds: float, width: int, fps: int) -> None:
    palette = out.with_suffix(".palette.png")
    base = ["-ss", str(start), "-t", str(seconds), "-i", str(video)]
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *base, "-vf", f"{scale},palettegen=max_colors=96", str(palette)],
                   check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *base, "-i", str(palette), "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", str(out)], check=True)
    palette.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hybrid-band", type=Path, required=True)
    parser.add_argument("--mot-video", type=Path, required=True)
    parser.add_argument("--frame-time", type=float, default=6.0)
    parser.add_argument("--gif-start", type=float, default=2.0)
    parser.add_argument("--gif-seconds", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    frame = extract_frame(args.mot_video, args.frame_time, args.out_dir / "_frame.png")
    (args.out_dir / "_frame.png").unlink(missing_ok=True)
    build_hero(Image.open(args.hybrid_band), frame, args.out_dir / "hero.png")
    build_social(frame, args.out_dir / "social_preview.png")
    if not args.no_gif:
        build_gif(args.mot_video, args.out_dir / "hero.gif", args.gif_start, args.gif_seconds, 960, 12)
    for name in ("hero.png", "social_preview.png", "hero.gif"):
        path = args.out_dir / name
        if path.exists():
            print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
