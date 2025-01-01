import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time
from scripts.process_modules.process_helpers import calc_ratio
# from scripts.process_modules.process_helpers import get_central_coordinate_from_tiff
# from scripts.process_modules.process_helpers import get_crs_from_tiff
# from preprocess_modules.organise_folders import rename_folder_based_on_country
# from preprocess_modules.organise_folders import get_country_name
# from preprocess_modules.organise_folders import update_asset_jsons
# from preprocess_modules.organise_folders import update_catalogue_json
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
    base_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_all_processing\TSX_TILES\TSX_TILES_COMPLETESETS\TSX_18")
    # print("---Processing dataset folders in:", base_path)
    # print(f"----Checking path: {base_path}")
    txtfile = base_path / "folder_names.txt"
    for folder in base_path.iterdir():
        # OPEN TXT FILE TO WRITE FOLDER NAMES
        # with open(txtfile, 'a') as f:
        #     print(f"---Processing folder: {folder.name}")
        #     f.write(f"{folder.name}\n")
        #     continue
        
        if folder.is_dir():
            print(f"---Processing folder: {folder.name}")

            tiles = list(f for f in folder.iterdir() if f.suffix == '.tif' )  # Looks for any file ending with img.tif
            num = len(tiles)

                
            if tiles:
                # Use the first TIFF file found
                # tiff_path = tiles[0]
                # Get the original folder name before renaming
                # old_folder_name = folder_path.name
                ratio = calc_ratio(folder)
                print(f"---old_folder_name: {folder.name}")
                # new_name =  folder.parent / f'{"_".join(folder.name.split(".")[:1])}_{num}tiles_r{ratio:.2f}'
                new_name =  folder.parent / f'{folder.name}_normalized'

                # rename the folder
                folder.rename(new_name)
                print(f"---new_name: {new_name.name}")
                # Proceed with renaming using the  TIFF file to get the CRS
                # new_folder_path = rename_folder_based_on_country(folder_path, tiff_path)

                # if new_folder_path:
                    # Update the STAC JSON file with the new folder name
                    # update_asset_jsons(old_folder_name, new_folder_path) 
                
            else:
                    print(f"---No suitable TIFF file found in {folder.name}") 
            

        
    # update_catalogue_json(old_folder_name, base_path, new_folder_path) 


base_path_root = r"X:\1NEW_DATA\1data\2interim"
base_path = Path(base_path_root) / "dataset_DLR_S1S2_bycountry"
# print("---Processing dataset folders in:", base_path)
# print(f"----Checking path: {base_path}")
if __name__ == "__main__":
    main(base_path)

