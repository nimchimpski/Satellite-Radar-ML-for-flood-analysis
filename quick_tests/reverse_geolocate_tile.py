
from pathlib import Path


from modules.location_crs_module import get_central_coordinate_from_tiff
from modules.location_crs_module import get_crs_from_tiff
from modules.location_crs_module import reverse_geolocate_country_utmzone
from modules.location_crs_module import reproject_to_epsg4326


base_path = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\val") 
'''
iterate through tiles, check if they have a crs in the metadata, if not, add it
'''         
def main(base_path):
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return
    for tile_path in base_path.iterdir():
        if tile_path.is_file():
            print(f"---Processing tile: {tile_path.name}")
            crs = get_crs_from_tiff(tile_path)
            if not crs:
                print(f'---No CRS found in Tiff for {tile_path}')
                lat, long = get_central_coordinate_from_tiff(tile_path) 
                print('---lat long:', lat, long)
                country, utmcrs = reverse_geolocate_country_utmzone(lat, long)
                if utmcrs:
                    print('---utmcrs:', utmcrs)
                else:
                    print(f'---No UTM CRS found for {tile_path}')

                # Step 3: Reproject the tile to EPSG:4326
                reproject_to_epsg4326(tile_path, utmcrs)

if __name__ == "__main__":
    main(base_path)
                