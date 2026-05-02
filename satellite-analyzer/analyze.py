import sys
import os

import cv2
import numpy as np


def pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.mean())


def main() -> int:
    # If no args: list local samples/ images and let user pick one.
    if len(sys.argv) == 1:
        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        imgs = []
        if os.path.isdir(samples_dir):
            for n in os.listdir(samples_dir):
                if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}:
                    imgs.append(os.path.join(samples_dir, n))
        imgs = sorted(imgs)
        if not imgs:
            print("Usage: python analyze.py image.jpg")
            return 2
        for i, p in enumerate(imgs, 1):
            print(f"{i:02d}. {os.path.basename(p)}")
        choice = input("Pick image number: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(imgs)):
            print("Invalid choice")
            return 2
        path = imgs[int(choice) - 1]
    elif len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("Usage: python analyze.py image.jpg")
        return 2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Error: could not read '{path}'")
        return 1

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Water: simple "blue-ish" threshold in HSV
    water = cv2.inRange(hsv, (90, 50, 30), (140, 255, 255)).astype(bool)
    # Vegetation: simple "green-ish" threshold in HSV
    veg = cv2.inRange(hsv, (35, 40, 30), (85, 255, 255)).astype(bool)
    # Land/urban: everything else
    land = ~(water | veg)

    # Color-coded output (BGR): water=blue, vegetation=green, land=gray
    out = np.zeros_like(bgr)
    out[water] = (255, 0, 0)
    out[veg] = (0, 255, 0)
    out[land] = (140, 140, 140)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.jpg")
    cv2.imwrite(out_path, out)

    w, v, l = pct(water), pct(veg), pct(land)
    print(f"Water: {w:.0f}%  |  Vegetation: {v:.0f}%  |  Land: {l:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
