import os
import re
import requests
import pystac
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')
os.makedirs(GEOTIFF_DIR, exist_ok=True)

STAC_BASE = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-datetime"

stac_id = "CAPELLA_C13_SP_GEO_HH_20250830031430_20250830031509"

match = re.search(r'(\d{4})(\d{2})(\d{2})', stac_id)
yyyy, mm, dd = match.group(1), match.group(2), match.group(3)

stac_url = (
    f"{STAC_BASE}/"
    f"capella-open-data-{yyyy}/"
    f"capella-open-data-{yyyy}-{mm}/"
    f"capella-open-data-{yyyy}-{mm}-{dd}/"
    f"{stac_id}/{stac_id}.json"
)

print(f"Loading STAC item: {stac_url}")
item = pystac.Item.from_file(stac_url)

print("\nAvailable assets:")
for key, asset in item.assets.items():
    print(f"  {key}: {asset.href}")

asset_key = next((k for k in item.assets if k in ['HH', 'image', 'data']), list(item.assets.keys())[0])
geotiff_href = item.assets[asset_key].href
print(f"\nDownloading asset '{asset_key}': {geotiff_href}")

save_path = os.path.join(GEOTIFF_DIR, f"{stac_id}.tif")

if os.path.exists(save_path):
    print(f"Already downloaded, skipping: {save_path}")
else:
    response = requests.get(geotiff_href, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"\nSaved to: {save_path}")

# --- Inspect with Rasterio ---
print("\n--- Rasterio Metadata ---")
with rasterio.open(save_path) as src:
    print(f"  Shape      : {src.height} x {src.width}")
    print(f"  Bands      : {src.count}")
    print(f"  Dtype      : {src.dtypes}")
    print(f"  CRS        : {src.crs}")
    print(f"  Resolution : {src.res}")
    print(f"  Bounds     : {src.bounds}")
    print(f"  Transform  : {src.transform}")

    # Read full res for stats (still manageable as float32 single band)
    data = src.read(1).astype(np.float32)

    # Read downsampled for display
    scale = 1024 / src.width
    out_h = int(src.height * scale)
    display_data = src.read(
        1,
        out_shape=(out_h, 1024),
        resampling=Resampling.average
    ).astype(np.float32)

print("\n--- Array Stats ---")
print(f"  Min  : {data.min():.4f}")
print(f"  Max  : {data.max():.4f}")
print(f"  Mean : {data.mean():.4f}")
print(f"  Std  : {data.std():.4f}")

# --- Visualize (downsampled) ---
p2, p98 = np.percentile(display_data[display_data > 0], 2), np.percentile(display_data[display_data > 0], 98)
display = np.clip(display_data, p2, p98)
display = (display - p2) / (p98 - p2)

plt.figure(figsize=(10, 10))
plt.imshow(display, cmap='gray')
plt.colorbar(label='Normalized Intensity')
plt.title(f'Capella SAR GEO — {yyyy}-{mm}-{dd} (Los Angeles)\n{src.width}x{src.height} px | {src.res[0]:.4f}m resolution')
plt.axis('off')
plt.tight_layout()

preview_path = os.path.join(DATA_DIR, 'samples', 'geotiff_preview.png')
plt.savefig(preview_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Preview saved to: {preview_path}")