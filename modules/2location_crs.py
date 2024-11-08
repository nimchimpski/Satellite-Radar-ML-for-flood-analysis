import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
from pathlib import Path
import json
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling


    # Define target CRS as EPSG:4326
    dst_crs = CRS.from_epsg(4326)  # WGS84 EPSG:4326
    
    # Calculate transform and new dimensions for the reprojected dataset
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    
    # Update metadata for the new file
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    # Define output path if not provided
    if output_path is None:
        input_path = Path(src.name)  # Original file path
        output_path = input_path.with_name(f"epsg4326_{input_path.name}")

    # Reproject and save to the new file
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

    return output_path  # Return the path of the reprojected file