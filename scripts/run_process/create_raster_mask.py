import rasterio
import numpy as np
from pathlib import Path

def create_raster_mask(mask_path, output_raster, no_data_value=None):
    # Load the mask file
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs

        # Identify no-data value
        if no_data_value is None:
            no_data_value = src.nodata
        if no_data_value is None:
            raise ValueError("No no-data value found in metadata or provided.")

        # Create a binary mask (1 for valid data, 0 for no-data)
        binary_mask = (mask != no_data_value).astype(np.uint8)

    # Save the binary mask as a GeoTIFF
    with rasterio.open(
        output_raster,
        "w",
        driver="GTiff",
        height=binary_mask.shape[0],
        width=binary_mask.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(binary_mask, 1)

    print(f"Raster mask saved to {output_raster}")

# Example usage
mask_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_all_processing\TSX_TO_PROCESS_0\695972138_5_MASK.tif")
output_raster = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_all_processing\TSX_TO_PROCESS_0\extent.tif")
create_raster_mask(mask_path, output_raster)
