import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time

base_path = Path(r"X:\1NEW_DATA\1data\2interim") / "dataset_DLR_S1S2_bycountry"
def rename(base_path)
    '''
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

rename(base_path)   