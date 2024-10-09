import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time


def rename(base_path)
    '''
    Remove words or characters etc from folder names
    base_path is a folder of folders
    '''
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            print(f"---Processing folder: {folder_path.name}")
            if 'Unknown' in folder_path.name:
                # delete the work 'unknown' in the folder name
                new_folder_name = folder_path.name.replace('Unknown', '')
                new_folder_name = folder_path.name.replace('Unknown_', '')
                new_folder_path = folder_path.parent / new_folder_name  

                                # Rename the folder
                print(f"Renaming: {folder_path} -> {new_folder_path}")
                folder_path.rename(new_folder_path)

# Function to get the central coordinates from a TIFF file
def get_central_coordinate_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        # Get the bounding box of the dataset
        bbox = dataset.bounds
        # Calculate the central coordinate in the native CRS
        central_x = (bbox.left + bbox.right) / 2
        central_y = (bbox.bottom + bbox.top) / 2
        # Get the dataset CRS (e.g., UTM)
        src_crs = dataset.crs
        
        # Transform to WGS84 (EPSG:4326) for reverse geocoding (lat/lon)
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        central_lon, central_lat = transformer.transform(central_x, central_y)
        
        return central_lat, central_lon
    
def get_crs_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        if not dataset.crs:
            raise ValueError("---CRS not found in the TIFF file")
        print(f"---CRS: {dataset.crs}")
        return dataset.crs 