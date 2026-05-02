# satellite-analyzer

Tiny OpenCV + NumPy script that classifies pixels into:
- Water (blue-ish)
- Vegetation (green-ish)
- Land/urban (everything else)

It saves a color-coded `result.jpg` and prints a simple percentage summary.

## Run (4 commands)

```bash
git clone <repo>
cd satellite-analyzer
pip install -r requirements.txt
python analyze.py image.jpg
```

## Quick test

```bash
python analyze.py samples/water.jpg
```

## Notes
- Thresholds are intentionally simple HSV ranges (no ML).
- Works best on images where water/vegetation are visually blue/green.
