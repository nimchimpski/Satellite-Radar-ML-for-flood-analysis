
from pathlib import Path
from osgeo import gdal
import sys
# module_path = Path(r"Z:\1NEW_DATA\3scripts\data-preprocessing\modules")
# # Add this path to Python's search path (sys.path)
# sys.path.append(str(module_path))

from modules.location_crs import get_central_coordinate_from_tiff
from modules.location_crs import get_crs_from_tiff
from modules.location_crs import reverse_geolocate_country_utmzone
from modules.location_crs import reproject_to_epsg4326

base_path = Path(r"Z:\1NEW_DATA\1data\2interim\all dlr waterset\dataset_DLR_S1S2_bycountry_utmz_copy") 
'''
iterate through tiles, check if they have a crs in the metadata, if not, add it
'''         
def reproject_tifs_to_4326(base_path):
    print('++++IN REPROJECT TIFS TO 4326')
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return  
    processed = 0
    for event in base_path.iterdir():
        if event.is_dir():
            for folder in event.iterdir():
                print(f'---at folder {folder.name}')
                if folder.is_dir() and folder.name == 'datacube_files':
                    # print(f'---{event.name} {folder.name} found')
                    for file in folder.iterdir():
                        print(f'---processing {file.name}')
                        # Only process files with a .tif or .tiff extension
                        if file.suffix.lower() not in ['.tif', '.tiff'] or 'IMAGE' in file.name:
                            print(f"---Skipping non-TIFF file: {file.name}")
                            continue
                        # Try opening the file with GDAL to check if it's a TIFF
                        dataset = gdal.Open(str(file))
                        if dataset and dataset.GetDriver().ShortName != 'GTiff':
                            print(f"{file} is not a valid TIFF file")
                            continue
                        # else:
                            # print(f"{file.name} is a valid TIFF file")
 
                        oldcrs = get_crs_from_tiff(file)
                        if oldcrs and '4326' in oldcrs:
                            print(f'---{file.name} crs was already 4326 ')
                            continue                  
                        if not oldcrs:
                            print(f'---No CRS found in Tiff for {file}, reverse geolocating')
                            lat, long = get_central_coordinate_from_tiff(file)
                            oldcrs = reverse_geolocate_country_utmzone(lat, long)
                        processed += 1
                        # Step 3: Reproject the tile to EPSG:4326
                        reproject_to_epsg4326(file, oldcrs)
                        print(f'---{file.name} processed')
    print(f'---{processed} files processed')

