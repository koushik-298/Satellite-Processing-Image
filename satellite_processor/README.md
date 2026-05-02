# Satellite Processor (Simple MVP)

Small, no-ML image processing CLI for satellite imagery.

## Setup

```bash
pip install -r satellite_processor/requirements.txt
```

## Usage

```bash
python satellite_processor/main.py --input satellite_processor/samples/sample.jpg --operation grayscale
python satellite_processor/main.py --input satellite_processor/samples/sample.jpg --operation edges
python satellite_processor/main.py --input satellite_processor/samples/sample.jpg --operation enhance
python satellite_processor/main.py --input sample.tif --operation ndvi
python satellite_processor/main.py --input satellite_processor/samples/sample.jpg --operation all
```

Outputs are written to `satellite_processor/output/` by default.

## Operations

- `grayscale`: grayscale conversion
- `enhance`: contrast enhancement (CLAHE)
- `edges`: Canny edge detection
- `ndvi`: NDVI-like visualization (works best with 4-band images; for RGB uses a pseudo-NDVI)
