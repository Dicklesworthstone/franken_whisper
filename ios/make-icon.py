#!/usr/bin/env python3
"""Draw the FrankenWhisper app icon (1024x1024 PNG) into the asset catalog.

Same visual family as the sibling FrankenSuite apps: the site's emerald
gradient (#04351f -> #34d399) with a stitched-waveform mark — a mono waveform
whose center bar is a lightning bolt of the lab's emerald, plus the two
corner bolts. Pure PIL; regenerate with: python3 ios/make-icon.py
"""

import math
import os

from PIL import Image, ImageDraw

SIZE = 1024
OUT = os.path.join(
    os.path.dirname(__file__), "Assets.xcassets", "AppIcon.appiconset", "icon-1024.png"
)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def main():
    img = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)

    # Diagonal gradient wash, #04351f -> #34d399 at ~140deg like the site.
    top = (2, 10, 6)
    lo = (4, 53, 31)
    hi = (52, 211, 153)
    for y in range(SIZE):
        for_x_t = y / SIZE
        row = lerp(lo, hi, for_x_t * 0.55)
        base = lerp(top, row, 0.85)
        draw.line([(0, y), (SIZE, y)], fill=base)

    # Waveform: symmetric envelope bars across the middle.
    mid = SIZE // 2
    bars = 17
    span = SIZE * 0.72
    x0 = (SIZE - span) / 2
    step = span / (bars - 1)
    heights = [
        0.18, 0.30, 0.24, 0.48, 0.36, 0.62, 0.44, 0.82, 1.00,
        0.82, 0.44, 0.62, 0.36, 0.48, 0.24, 0.30, 0.18,
    ]
    max_h = SIZE * 0.33
    bar_w = step * 0.42
    for i, h in enumerate(heights):
        if i == bars // 2:
            continue  # the center slot belongs to the bolt
        cx = x0 + i * step
        half = max_h * h
        col = lerp((6, 24, 16), (230, 255, 244), 0.25 + 0.75 * h)
        draw.rounded_rectangle(
            [cx - bar_w / 2, mid - half, cx + bar_w / 2, mid + half],
            radius=bar_w / 2,
            fill=col,
        )

    # Center lightning bolt in bright emerald with a dark outline.
    cx = x0 + (bars // 2) * step
    h = max_h * 1.4
    w = step * 3.4
    bolt = [
        (cx + w * 0.16, mid - h),
        (cx - w * 0.34, mid + h * 0.10),
        (cx - w * 0.04, mid + h * 0.10),
        (cx - w * 0.16, mid + h),
        (cx + w * 0.34, mid - h * 0.10),
        (cx + w * 0.04, mid - h * 0.10),
    ]
    draw.polygon(bolt, fill=(52, 211, 153), outline=(2, 20, 13), width=6)

    # Corner bolt studs (top-left, bottom-right), the lab's signature.
    for bx, by in [(SIZE * 0.11, SIZE * 0.11), (SIZE * 0.89, SIZE * 0.89)]:
        r = SIZE * 0.045
        draw.ellipse([bx - r, by - r, bx + r, by + r], fill=(28, 34, 32),
                     outline=(90, 110, 100), width=6)
        for angle in (45, -45):
            dx = math.cos(math.radians(angle)) * r * 0.62
            dy = math.sin(math.radians(angle)) * r * 0.62
            draw.line([bx - dx, by - dy, bx + dx, by + dy], fill=(70, 84, 78), width=10)

    img.save(OUT, "PNG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
