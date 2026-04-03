import pystac
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from urllib.parse import urljoin
import csv
import pandas as pd
from datetime import datetime, timezone
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor,as_completed
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    
def process_item(item_link):
    """Process a single item and extract key parameters"""
    try:
        item = pystac.Item.from_file(item_link)
        props = item.properties
        
        # Get centroid from proj:centroid if available
        centroid = props.get('proj:centroid', {})
        
        return {
            'stac_id': item.id,
            'collect_id': props.get('capella:collect_id', ''),
            'datetime': props.get('datetime', ''),
            'start_datetime': props.get('start_datetime', ''),
            'end_datetime': props.get('end_datetime', ''),
            'center_lon': centroid.get('lon', ''),
            'center_lat': centroid.get('lat', ''),
            'platform': props.get('platform', ''),
            'constellation': props.get('constellation', ''),
            'instrument_mode': props.get('sar:instrument_mode', ''),
            'frequency_band': props.get('sar:frequency_band', ''),
            'center_frequency': props.get('sar:center_frequency', ''),
            'polarizations': ','.join(props.get('sar:polarizations', [])),
            'orbit_state': props.get('sat:orbit_state', ''),
            'product_type': props.get('sar:product_type', ''),
            'observation_direction': props.get('sar:observation_direction', ''),
            'incidence_angle': props.get('view:incidence_angle', ''),
            'azimuth': props.get('view:azimuth', ''),
            'squint_angle': props.get('capella:squint_angle', ''),
            'layover_angle': props.get('capella:layover_angle', ''),
            'look_angle': props.get('capella:look_angle', ''),
            'resolution_range': props.get('sar:resolution_range', ''),
            'resolution_azimuth': props.get('sar:resolution_azimuth', ''),
            'resolution_ground_range': props.get('capella:resolution_ground_range', ''),
            'pixel_spacing_range': props.get('sar:pixel_spacing_range', ''),
            'pixel_spacing_azimuth': props.get('sar:pixel_spacing_azimuth', ''),
            'image_length': props.get('capella:image_length', ''),
            'image_width': props.get('capella:image_width', ''),
            'looks_range': props.get('sar:looks_range', ''),
            'looks_azimuth': props.get('sar:looks_azimuth', ''),
            'orbital_plane': props.get('capella:orbital_plane', ''),
            'collection_type': props.get('capella:collection_type', ''),
        }
    except Exception as e:
        print(f"Error processing {item_link}: {e}")
        return None
    
    
# Open collection
print("Opening collection...")
collection_url = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"
collection = pystac.Collection.from_file(collection_url)

# Get item links and resolve relative URLs
print("Collecting item links...")
item_links = [link.absolute_href for link in collection.get_item_links()]
# for link in collection.get_item_links():
#     if link.rel == 'item':
#         # Resolve relative path to absolute URL
#         if link.href.startswith('./'):
#             # Remove './' and join with base URL
#             absolute_url = urljoin(collection_url, link.href[2:])
#             item_links.append(absolute_url)
#         elif link.href.startswith('http'):
#             # Already absolute
#             item_links.append(link.absolute_href)
#         else:
#             # Relative without './'
#             absolute_url = urljoin(collection_url, link.absolute_href)
#             item_links.append(absolute_url)

# print(f"Found {len(item_links)} items")
# print(f"Example URL: {item_links[0]}")  # Check if it looks right

print(f"Processing with 20 threads...")
results = []

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_item, link) for link in item_links]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result:
            results.append(result)
            
print(f"Successfully processed {len(results)} items")


# filename = "../data/"
# if results:
#     with open(filename, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)
#     print(f"Saved to {filename}")
    
filename = os.path.join(DATA_DIR, 'explore_dataset.csv')

if not os.path.exists(filename):
    with open(filename, 'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {filename}")
else:
    print(f"The file '{filename}' already exists.")
    
    
    

# Load the CSV
df = pd.read_csv(filename)

# Convert datetime columns to pandas datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['start_datetime'] = pd.to_datetime(df['start_datetime'])
df['end_datetime'] = pd.to_datetime(df['end_datetime'])

# Define time range
start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
end_time = datetime.now(timezone.utc)

# Define Los Angeles bounding box (approximate)
# LA is roughly at 34.05°N, -118.24°W
la_bbox = {
    'west': -118.7,
    'east': -117.7,
    'south': 33.7,
    'north': 34.4
}

# Filter by time and location
filtered_df = df[
    (df['start_datetime'] >= start_time) &
    (df['start_datetime'] <= end_time) &
    (df['center_lon'] >= la_bbox['west']) &
    (df['center_lon'] <= la_bbox['east']) &
    (df['center_lat'] >= la_bbox['south']) &
    (df['center_lat'] <= la_bbox['north'])
]

print(f"Found {len(filtered_df)} items over Los Angeles")
print(f"Date range: {filtered_df['start_datetime'].min()} to {filtered_df['start_datetime'].max()}")



print("\nFiltered items:")
print(tabulate(
    filtered_df[['stac_id', 'datetime', 'center_lon', 'center_lat', 'instrument_mode']],
    headers='keys',
    tablefmt='grid',
    showindex=False
))
# Save to new CSV
filename_filtered = os.path.join(DATA_DIR, 'explore_dataset_filtered.csv')
filtered_df.to_csv(f"{filename_filtered}", index=False)
print(f"\nSaved to {filename_filtered}")