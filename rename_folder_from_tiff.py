import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time
'''
iterates through a folder and subfolders to find a tiff file and rename the folder based on the country name.
INDIVIDUAL FILE NAMES ARE NOT CHANGED, JUST THE FOLDER NAME AND THE PATHS INSIDE THE JSON FILES.
'''

# Initialize the geolocator for reverse geocoding
geolocator = Nominatim(user_agent="floodai")

def get_crs_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        if not dataset.crs:
            raise ValueError("---CRS not found in the TIFF file")
        print(f"---CRS: {dataset.crs}")
        return dataset.crs      

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

# Function to get the country name using reverse geocoding
def get_country_name(lat, lon):
    try:
        # print(f"---Reverse geocoding lat: {lat}, lon: {lon}")  # Debug: Print coordinates being geocoded
        time.sleep(1)  # Add a delay between requests
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'country' in location.raw['address']:
            return location.raw['address']['country']
        return "Unknown"
    except Exception as e:
        print(f"---Error during reverse geocoding: {e}")
        return "Unknown"

# Function to rename the folder based on the country name
def rename_folder_based_on_country(folder_path, tiff_path):
    # Get the CRS from the TIFF file
    tiff_crs = get_crs_from_tiff(tiff_path)

    # Get the central coordinate of the tiff file using the CRS from the TIFF
    central_lat, central_lon = get_central_coordinate_from_tiff(tiff_path)
    
    # Get the country name from the central coordinates
    country = get_country_name(central_lat, central_lon)
    print(f"---Country: {country} for folder: {folder_path}")
    
    # Define the new folder name
    new_folder_name = f"{country}_{folder_path.name}"
    new_folder_path = folder_path.parent / new_folder_name
    
    # Rename the folder
    try:
        os.rename(folder_path, new_folder_path)
        # print(f"Renamed folder from {folder_path} to {new_folder_path}")
        return new_folder_path
    except Exception as e:
        print(f"---Error renaming folder: {e}")
        return None

def update_asset_jsons(old_folder_name, new_folder_path):
    """
    Update the asset JSON files in the renamed folder to reflect the new folder name.

    :param old_folder_name: The original folder name before renaming.
    :param new_folder_path: Path to the folder containing the renamed assets.
    """
    # Iterate over all JSON files in the folder that might contain assets
    for json_file in new_folder_path.glob("*.json"):
        # Skip the catalog.json since it is handled separately
        if json_file.name == "catalog.json":
            continue
        
        with open(json_file, 'r') as f:
            asset_data = json.load(f)

        # Update the href links in the assets section
        updated = False
        if 'assets' in asset_data:
            for asset_key, asset_info in asset_data['assets'].items():
                if 'href' in asset_info:
                    old_href = asset_info['href']
                    # Replace old folder name with the new one
                    new_href = old_href.replace(str(old_folder_name), str(new_folder_path.name))
                    asset_info['href'] = new_href
                    updated = True
                    print(f"Updated asset href in {json_file.name} for {asset_key}: {old_href} -> {new_href}")

        # If there were any updates, save the modified JSON
        if updated:
            with open(json_file, 'w') as f:
                json.dump(asset_data, f, indent=4)
                print(f"---Updated asset JSON: {json_file.name}")

# function to update paths inside the stac jsonfile to reflect the new filder name   
def update_catalogue_json(old_folder_name, base_path, new_folder_path):
    """
    Update the paths inside the STAC JSON CATALOGUE file to reflect the new folder name.
    
     :param old_folder_name: The original folder name before renaming.
    :param new_folder_path: Path to the folder containing the renamed STAC JSON and assets.
    """
    print(f"+++++++Updating STAC CAT in {base_path}")
    stac_catalogue_path = Path(base_path) / "catalog.json"
    print(f"+++++++STAC CAT path: {stac_catalogue_path}")
    
    if stac_catalogue_path.exists():
        with stac_catalogue_path.open('r') as f:
            stac_data = json.load(f)
        
        # Update the asset links to reflect the new folder name
        for link in stac_data.get('links', []):
            if 'href' in link:
                print(f"---href: {link['href']}")
                old_href_path = Path(link['href'])
                folder_path = old_href_path.parent
                file_name = old_href_path.name    

                if f'/{old_folder_name}/' in str(folder_path):
                    new_folder_path_part = folder_path.as_posix().replace(f'/{old_folder_name}/', f'/{new_folder_path.name}/')
                    new_href = Path(new_folder_path_part) / file_name
                    link['href'] = str(new_href)

                    # Logging updated path
                    print(f"Updated asset path: {old_href_path} -> {new_href}")    

    # Save the updated catalog JSON back to file
    with stac_catalogue_path.open('w') as f:
        json.dump(stac_data, f, indent=4)
        print(f"---Updated STAC CATALOGUE JSON: {stac_catalogue_path}")


# Process dataset folders: Renaming and STAC JSON updates
def process_dataset_folders(base_path):
    """
    Traverse dataset folders and update their STAC JSON files with new folder names.

    :param base_path: Base path where the renamed folders are located.
    """

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
if base_path.exists():
    process_dataset_folders(base_path)
else:
    print("---base_Path does not exist:", base_path)

