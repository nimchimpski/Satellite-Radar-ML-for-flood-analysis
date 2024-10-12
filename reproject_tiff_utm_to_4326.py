
from pathlib import Path
from osgeo import gdal
import sys
module_path = Path(r"Z:\1NEW_DATA\3scripts\data-preprocessing\modules")
# Add this path to Python's search path (sys.path)
sys.path.append(str(module_path))

from location_crs_module import get_central_coordinate_from_tiff
from location_crs_module import get_crs_from_tiff
from location_crs_module import reverse_geolocate_country_utmzone
from location_crs_module import reproject_to_epsg4326

base_path = Path(r"Z:\1NEW_DATA\1data\2interim\all dlr waterset\dataset_DLR_S1S2_bycountry_utmz_copy") 
'''
iterate through tiles, check if they have a crs in the metadata, if not, add it
'''         
def main(base_path):
    print('---running main')
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return  
    processed = 0
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                # Only process files with a .tif or .tiff extension
                    if file_path.suffix.lower() not in ['.tif', '.tiff']:
                        # print(f"---Skipping non-TIFF file: {file_path.name}")
                        continue
                    # Try opening the file with GDAL to check if it's a TIFF
                    dataset = gdal.Open(str(file_path))
                    if dataset and dataset.GetDriver().ShortName != 'GTiff':
                        print(f"{file_path} is not a valid TIFF file")
                        continue
                    # print(f"{file_path.name} is a valid TIFF file")

                    oldcrs = get_crs_from_tiff(file_path)
                    if oldcrs and '4326' in oldcrs:
                        print(f'---crs was already 4326 {file_path}')
                        continue                  
                    if not oldcrs:
                        print(f'---No CRS found in Tiff for {file_path}')
                        lat, long = get_central_coordinate_from_tiff(file_path)
                        oldcrs = reverse_geolocate_country_utmzone(lat, long)
                    processed += 1
                    # Step 3: Reproject the tile to EPSG:4326
                    reproject_to_epsg4326(file_path, oldcrs)
            print(f'---{folder_path.name} processed')
        print(f'---{processed} files processed')
if __name__ == "__main__":
    main(base_path)
            