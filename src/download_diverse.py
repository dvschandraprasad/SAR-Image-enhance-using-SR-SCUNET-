import os
import re
import requests
import pystac
import pandas as pd
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
GEOTIFF_DIR = os.path.join(DATA_DIR, 'samples', 'geotiffs')
os.makedirs(GEOTIFF_DIR, exist_ok=True)

STAC_BASE  = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-datetime"
N_IMAGES   = 25   # total to download

# ── Load catalog CSV ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, 'explore_dataset.csv'))
geo_df = df[df['product_type'] == 'GEO'].copy()
print(f"Total GEO items: {len(geo_df)}")

# ── Diverse spatial sampling ───────────────────────────────────────────────────
# Bin the world into a grid and pick ~1 image per cell so we get
# geographic spread rather than a cluster over one city.
geo_df['lat_bin'] = pd.cut(geo_df['center_lat'], bins=10, labels=False)
geo_df['lon_bin'] = pd.cut(geo_df['center_lon'], bins=10, labels=False)
geo_df['cell']    = geo_df['lat_bin'].astype(str) + '_' + geo_df['lon_bin'].astype(str)

# Sample up to N_IMAGES, at most 1 per cell first, then fill if needed
sampled = (
    geo_df.groupby('cell', group_keys=False)
          .apply(lambda g: g.sample(1, random_state=42))
)
if len(sampled) < N_IMAGES:
    remaining = geo_df[~geo_df.index.isin(sampled.index)]
    extra     = remaining.sample(min(N_IMAGES - len(sampled), len(remaining)), random_state=42)
    sampled   = pd.concat([sampled, extra])

sampled = sampled.head(N_IMAGES).reset_index(drop=True)
print(f"Selected {len(sampled)} diverse GEO items")
print(sampled[['stac_id', 'center_lat', 'center_lon']].to_string())


def build_stac_url(stac_id):
    match = re.search(r'(\d{4})(\d{2})(\d{2})', stac_id)
    if not match:
        return None
    yyyy, mm, dd = match.group(1), match.group(2), match.group(3)
    return (
        f"{STAC_BASE}/"
        f"capella-open-data-{yyyy}/"
        f"capella-open-data-{yyyy}-{mm}/"
        f"capella-open-data-{yyyy}-{mm}-{dd}/"
        f"{stac_id}/{stac_id}.json"
    )


def download_geotiff(stac_id, save_path):
    stac_url = build_stac_url(stac_id)
    if not stac_url:
        print(f"  Could not parse date from {stac_id}, skipping.")
        return False

    try:
        item = pystac.Item.from_file(stac_url)
    except Exception as e:
        print(f"  Failed to load STAC item: {e}")
        return False

    asset_key = next((k for k in item.assets if k in ['HH', 'image', 'data']), list(item.assets.keys())[0])
    href = item.assets[asset_key].href

    try:
        response = requests.get(href, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=stac_id[-20:]) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


# ── Download loop ──────────────────────────────────────────────────────────────
success, failed = 0, []

for _, row in sampled.iterrows():
    stac_id   = row['stac_id']
    save_path = os.path.join(GEOTIFF_DIR, f"{stac_id}.tif")

    if os.path.exists(save_path):
        print(f"Already exists, skipping: {stac_id}")
        success += 1
        continue

    print(f"\nDownloading [{success+1}/{len(sampled)}]: {stac_id}")
    ok = download_geotiff(stac_id, save_path)
    if ok:
        success += 1
    else:
        failed.append(stac_id)

print(f"\nDone. {success}/{len(sampled)} downloaded successfully.")
if failed:
    print(f"Failed ({len(failed)}):")
    for f in failed:
        print(f"  {f}")