import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time
from scripts.preprocess_modules.preprocess_helpers import get_central_coordinate_from_tiff
from scripts.preprocess_modules.preprocess_helpers import get_crs_from_tiff
from preprocess_modules.organise_folders import rename_folder_based_on_country
from preprocess_modules.organise_folders import get_country_name
from preprocess_modules.organise_folders import update_asset_jsons
from preprocess_modules.organise_folders import update_catalogue_json
'''
Countryname grabbed from Nominatim.
Iterates through a folder and subfolders to find a tiff file and rename the folder based on the country name.
INDIVIDUAL FILE NAMES ARE NOT CHANGED, JUST THE FOLDER NAME AND THE PATHS INSIDE THE JSON FILES.
'''

# Initialize the geolocator for reverse geocoding
geolocator = Nominatim(user_agent="floodai")

# Process dataset folders: Renaming and STAC JSON updates
def main(base_path):
    """
    Traverse dataset folders and update their STAC JSON files with new folder names.

    :param base_path: Base path where the renamed folders are located.
    """
    base_path_root = r"X:\1NEW_DATA\1data\2interim"
    base_path = Path(base_path_root) / "dataset_DLR_S1S2_bycountry"
    # print("---Processing dataset folders in:", base_path)
    # print(f"----Checking path: {base_path}")

    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            print(f"---Processing folder: {folder_path}")

            tiff_files = list(folder_path.glob("*img.tif"))  # Looks for any file ending with img.tif
                
            if tiff_files:
                # Use the first TIFF file found
                tiff_path = tiff_files[0]
                # Get the original folder name before renaming
                old_folder_name = folder_path.name
                print(f"---old_folder_name: {old_folder_name}")
                # Proceed with renaming using the  TIFF file to get the CRS
                new_folder_path = rename_folder_based_on_country(folder_path, tiff_path)
                print(f"---new_folder_path: {new_folder_path}")

                if new_folder_path:
                    # Update the STAC JSON file with the new folder name
                    update_asset_jsons(old_folder_name, new_folder_path) 

            else:
                    print(f"---No suitable TIFF file found in {folder_path}") 
        
    update_catalogue_json(old_folder_name, base_path, new_folder_path) 


base_path_root = r"X:\1NEW_DATA\1data\2interim"
base_path = Path(base_path_root) / "dataset_DLR_S1S2_bycountry"
# print("---Processing dataset folders in:", base_path)
# print(f"----Checking path: {base_path}")
if __name__ == "__main__":
    main(base_path)

