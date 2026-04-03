import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
THUMBNAILS_DIR = os.path.join(DATA_DIR, 'samples', 'thumbnails')

files = [f for f in os.listdir(THUMBNAILS_DIR) if f.endswith('.png')]
files.sort()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, fname in enumerate(files):
    img = mpimg.imread(os.path.join(THUMBNAILS_DIR, fname))
    axes[i].imshow(img, cmap='gray')
    # Show just the date portion from the filename
    parts = fname.replace('.png', '').split('_')
    date_str = parts[5][:8] if len(parts) > 5 else fname
    axes[i].set_title(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}", fontsize=12)
    axes[i].axis('off')

# Hide any unused subplots
for j in range(len(files), len(axes)):
    axes[j].axis('off')

plt.suptitle('Capella SAR GEO Thumbnails — Los Angeles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()