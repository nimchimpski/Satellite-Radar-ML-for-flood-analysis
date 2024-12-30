import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm

# Assume tiles is a list of numpy arrays (ground truth masks)
# Each tile has 1 for flooded and 0 for non-flooded pixels
flooded_count = 0
non_flooded_count = 0

tiles = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_all_processing\TSX_TILES\HOLD_BACK\695971749_2.nc_normalized_tiles_logclipmm_g_pcnf100" )
for tile in tqdm(tiles.iterdir(), total=len(list(tiles.iterdir()))):
    if tile.suffix != ".tif":
        continue
    with rasterio.open(tile) as src:
        data = src.read(3)
    flooded_count += np.sum(data == 1)
    non_flooded_count += np.sum(data == 0)


# Calculate class ratio
total_pixels = flooded_count + non_flooded_count
class_ratio = flooded_count / total_pixels
print(f'---event: {event.name}')
print(f"Flooded Class Ratio: {class_ratio:.2f}")
