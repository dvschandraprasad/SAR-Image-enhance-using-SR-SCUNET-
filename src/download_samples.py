import pandas as pd
import os
import re
import requests
import pystac


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'explore_dataset.csv'))

la_bbox = {
    'west': -118.7,
    'east': -117.7,
    'south': 33.7,
    'north': 34.4
}

geo_filter = df[
    (df['product_type'] == 'GEO') &
    (df['center_lon'] >= la_bbox['west']) &
    (df['center_lon'] <= la_bbox['east']) &
    (df['center_lat'] >= la_bbox['south']) &
    (df['center_lat'] <= la_bbox['north'])
]

print(f"Found {len(geo_filter)} GEO items over Los Angeles")

thumbnails_dir = os.path.join(DATA_DIR, 'samples', 'thumbnails')
os.makedirs(thumbnails_dir, exist_ok=True)

STAC_BASE = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-datetime"

for index, row in geo_filter.iterrows():
    stac_id = row['stac_id']
    print(f"\nProcessing: {stac_id}")

    # Step 1: Extract date components from stac_id
    match = re.search(r'(\d{4})(\d{2})(\d{2})', stac_id)
    if not match:
        print(f"  Could not parse date from stac_id, skipping.")
        continue
    yyyy, mm, dd = match.group(1), match.group(2), match.group(3)

    # Step 2: Reconstruct the STAC item JSON URL
    stac_url = (
        f"{STAC_BASE}/"
        f"capella-open-data-{yyyy}/"
        f"capella-open-data-{yyyy}-{mm}/"
        f"capella-open-data-{yyyy}-{mm}-{dd}/"
        f"{stac_id}/{stac_id}.json"
    )
    print(f"  STAC URL: {stac_url}")

    # Step 3: Load STAC item and get thumbnail href
    try:
        item = pystac.Item.from_file(stac_url)
        thumbnail_href = item.assets['thumbnail'].href
        print(f"  Thumbnail href: {thumbnail_href}")
    except Exception as e:
        print(f"  Failed to load STAC item: {e}")
        continue

    # Step 4: Download and save the thumbnail
    ext = os.path.splitext(thumbnail_href)[-1] or '.png'
    save_path = os.path.join(thumbnails_dir, f"{stac_id}{ext}")

    try:
        response = requests.get(thumbnail_href, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"  Saved to: {save_path}")
    except Exception as e:
        print(f"  Failed to download thumbnail: {e}")