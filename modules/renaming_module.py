import re
from pathlib import Path
import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time

def remove_characters_from_foldername(base_path):
    '''
    rename folders by removing everything apart from digits between 1 and 99
    '''
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return
    
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            print(f"---Processing folder: {folder_path.name}")
            
            # Use regex to find digits between 1 and 99
            digits = re.findall(r'\d{1,2}', folder_path.name)  # Matches 1 or 2 digit numbers
            
            # Filter out any numbers greater than 99 (in case there are such numbers)
            digits = [digit for digit in digits if 1 <= int(digit) <= 99]
            
            if digits:
                # Join the matched digits into a new folder name
                new_folder_name = '_'.join(digits)
                new_folder_path = folder_path.parent / new_folder_name
                
                # Check if the new folder already exists to avoid errors
                if not new_folder_path.exists():
                    # Rename the folder
                    print(f"Renaming: {folder_path} -> {new_folder_path}")
                    folder_path.rename(new_folder_path)
                else:
                    print(f"Folder with the name {new_folder_path} already exists. Skipping.")
            else:
                print(f"No digits between 1 and 99 found in {folder_path.name}. Skipping.")

def remove_word_from_foldername(base_path, word):

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
                new_folder_name = folder_path.name.replace(word, '')
                new_folder_path = folder_path.parent / new_folder_name  

                                # Rename the folder
                print(f"Renaming: {folder_path} -> {new_folder_path}")
                folder_path.rename(new_folder_path)

def remove_word_from_filename(file_path, word):
    '''
    Remove words or characters etc from folder names
    base_path is a folder of folders
    '''
    if file_path.is_file():
        if word in file_path.name:
            new_folder_name = file_path.name.replace(word, '')
            new_file_path = file_path.parent / new_folder_name 
            print(f"Renaming: {file_path} -> {new_file_path}")
            file_path.rename(new_file_path)


    
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
