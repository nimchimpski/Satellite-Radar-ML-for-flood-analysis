import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import ColorInterp
import rioxarray as rxr 
from pathlib import Path
from tqdm import tqdm


def tile_datacube(datacube_path, save_tiles_path, tile_size=256, stride=256):
    """
    Tile a DATASET (extracted from a dataarray is selected) and save to 'tiles' dir in same location.
    'ARGs:
    
    play with the stride and tile_size to get the right size of tiles.
    TODO add date etc to saved tile name?
    TODO add json file with metadata for each tile inv mask and anal_ext pixel count etc
    """
    print('\n+++++++in tile_datacube fn ++++++++')

    def contains_nans(tile):
        """
        Checks if specified bands contain NaN values.
        Returns True if NaNs are found, along with the count of NaNs per band.
        Returns False if no NaNs are found.
        """
        bands_to_check = ["dem", "slope", "vv", "vh", 'valid','mask']
        contains_nan = False  # Flag to track if any NaNs are found

        for band in bands_to_check:
            # Calculate the number of NaNs in the current band
            nan_count = np.isnan(tile.sel(layer=band)).sum().values
            if nan_count > 0:
                print(f"Tile contains {nan_count} NaN values in {band}")
                contains_nan = True
        return contains_nan  # True if any NaNs were found, False if none

    def has_no_valid(tile):
        '''
        Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
        return False = ok
        '''
        print("---Layer values: ", tile.coords['layer'].values)
     
        valid_data = tile.sel(layer='valid')
        # print("---Valid layer content: ", valid_data)
        print("---Valid layer stats: min=", valid_data.min().item(), ", max=", valid_data.max().item())


        # Ensure 'valid' layer exists
        if 'valid' not in tile.coords['layer'].values:
            print("---Warning: 'valid' layer not found in tile.")
            return True  # If 'valid' layer is missing, treat as no valid data

        if 1 not in tile.sel(layer='valid'):
            print("---Warning: No valid data found in tile.")
            #print("Unique values in 'valid' layer:", np.unique(datacube.sel(layer='valid').values))
            #print("Values in 'valid' layer for tile:", tile.sel(layer='valid').values)
            return True  # If no valid data is found, treat as no valid data
        return False  # If valid data is found, return False
    
    def has_no_mask(tile):
        '''
        Filters out tiles without mask.
        return False = ok
        '''
        return 'mask' not in tile.coords['layer'].values

    # datacube = rxr.open_rasterio(datacube_path, variable='data1') # OPENS A DATAARRAY

    # EXTRACT THE DATAARRAY WE WANT
    datacube = xr.open_dataset(datacube_path) # OPENS A DATASET
    datacube = datacube['data1']

    print('==============================')
    print('---opened datacube= ', datacube)
    print('==============================')
    # CHECKING THE VALID LAYER
    # check if datacube is a dataarray or dataset
    if isinstance(datacube, xr.Dataset):
        print('---datacube is a dataset')
        # Select the 'valid' layer within 'data1'
        valid_values = datacube['data1'].sel(layer='valid').values
    else:
        print('---datacube is a dataarray')
        # datacube = datacube.to_array(dim='layer', name='data1')
        valid_values = datacube.sel(layer='valid').values

    print('---opened datacube crs1 = ', datacube.rio.crs)
    datacube.rio.write_crs("EPSG:4326", inplace=True)
    print('---opened datacube crs2 = ', datacube.rio.crs)


    # Check unique values in the 'valid' layer
    unique_valid_values = np.unique(valid_values)
    print("---Unique values in 'valid' layer for 'data1':", unique_valid_values)

    num_x_tiles = max(datacube.x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(datacube.y.size + stride - 1, 0) // stride + 1
    # num_x_tiles = 100
    # num_y_tiles = 100

    total_num_tiles = 0
    num_has_nans = 0
    num_nomask = 0
    num_saved = 0
    num_novalid = 0

    for y_start in tqdm(range(0, datacube.y.size, stride), desc="### Processing tiles by row"):
        for x_start in range(0, datacube.x.size, stride):
            # Ensure tiles start on the boundary and fit within the dataset
            x_end = min(x_start + tile_size, datacube.x.size)
            y_end = min(y_start + tile_size, datacube.y.size)

            # Select the subset of data for the current tile
            try:
                tile = datacube.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            except KeyError:
                print("---Error: tiling.")
                return

            # Skip empty tiles
            if tile.sizes["x"] == 0 or tile.sizes["y"] == 0:
                print("---Empty tile encountered; skipping.")
                continue
            print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
            print('---TILE= ', tile)
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

            #CREAE A JSON FILE FOR EACH TILE
            # create_tile_json(tile, event, x_idx, y_idx, tile_path)   

            total_num_tiles += 1

            if total_num_tiles % 250 == 0:
                print(f"Counted {total_num_tiles} tiles")

            # set the criterion here.
            if contains_nans(tile):
                num_has_nans += 1
                print('---tile has nans')
                continue

            if has_no_valid(tile):
                num_novalid += 1
                print('---tile has no valid')

                continue

            if has_no_mask(tile):
                num_nomask +=1
                print('---tile has no mask')
                continue

            # TODO mask pixel check here + add to json

            tile_name = f"tile_{datacube_path.parent.name}_{x_start}_{y_start}.tif"

            dest_path = save_tiles_path / datacube_path.name / tile_name
            if not dest_path.parent.exists():
                os.makedirs(dest_path.parent, exist_ok=True)
                print('---created dir= ', dest_path.parent)

            tile.rio.to_raster(dest_path, compress="deflate")
            num_saved += 1


        
    return total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask


