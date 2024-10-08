import rasterio
import numpy as np
from pathlib import Path

# Open the raster file
# tiff_path = "path_to_your_mask.tif"

path1 = Path(r"Z:\1NEW_DATA\1data\2interim\TSX_MASKS\Laos\902\2022-10-02T11-16-36\2022-10-02T11-16-36-flood_compressed.tif")
with rasterio.open(path1) as src:
    # Read the data from the first band
    band1 = src.read(1)

    # Check unique pixel values
    unique_values, counts = np.unique(band1, return_counts=True)
    print("Unique pixel values:", unique_values)
    print("Counts of each value:", counts)

