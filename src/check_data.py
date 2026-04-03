import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GEOTIFF_DIR = os.path.join(BASE_DIR, '..', 'data', 'samples', 'geotiffs')

files = sorted([f for f in os.listdir(GEOTIFF_DIR) if f.endswith('.tif')])

print(f"GeoTIFFs found: {len(files)}")
total_gb = 0
for f in files:
    path = os.path.join(GEOTIFF_DIR, f)
    gb   = os.path.getsize(path) / 1e9
    total_gb += gb
    print(f"  {f}  —  {gb:.2f} GB")

print(f"\nTotal size: {total_gb:.2f} GB")