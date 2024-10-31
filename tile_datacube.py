import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import ColorInterp
import rioxarray as rxr 
from pathlib import Path
from tqdm import tqdm

def tile_datacube(datacube, event, tile_size=256, stride=256):
    """
    Tile a datacube and save to 'tiles' dir in same location.
    'ARG event is a Path object
    play with the stride and tile_size to get the right size of tiles.
    TODO add date etc to saved tile name?
    """
    print('\n+++++++in tile_to_dir fn++++++++')

    def contains_nans(tile):
        """
        Checks if specified bands contain NaN values.
        Returns True if NaNs are found, along with the count of NaNs per band.
        Returns False if no NaNs are found.
        """
        bands_to_check = ["dem", "slope", "vv", "vh"]
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
        # Ensure 'valid' layer exists
        if 'valid' not in tile.coords['layer'].values:
            print("Warning: 'valid' layer not found in tile.")
            return True  # If 'valid' layer is missing, treat as no valid data

        return 1 not in tile.sel(layer='valid').values
    
    def has_no_mask(tile):
        '''
        Filters out tiles without mask.
        return False = ok
        '''
        return 'mask' not in tile.coords['layer'].values
    

    print('---making tiles dir  in event dir')
    print('---event= ', event)

    tile_path = Path(event, 'tiles') 
    os.makedirs(tile_path, exist_ok=True)
    # check new tiles dir exists
    print('---datacube= ', datacube)
    datacube = rxr.open_rasterio(datacube)
    # print datacube crs


    #print('---loaded datacube= ', datacube)
    print('---opened datacube crs = ', datacube.rio.crs)

    print('==============================')
    num_x_tiles = max(datacube.x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(datacube.y.size + stride - 1, 0) // stride + 1

    #num_x_tiles = 10
    #num_y_tiles = 10


    total_num_tiles = 0
    num_has_nans = 0
    num_nomask = 0
    num_saved = 0
    num_novalid = 0
    for y_idx in tqdm(range(num_y_tiles),desc="---Processing tiles by row"):

        for x_idx in range(num_x_tiles):

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, datacube.x.size)
            y_end = min(y_start + tile_size, datacube.y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            tile = datacube['__xarray_dataarray_variable__'].sel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            # print('--- 1 tile as dataarray before saving= ', tile)

            total_num_tiles += 1

            if total_num_tiles % 250 == 0:
                print(f"Counted {total_num_tiles} tiles")

            # set the criterion here.
            if contains_nans(tile):
                num_has_nans += 1
                continue

            if has_no_valid(tile):
                num_novalid += 1
                continue

            if has_no_mask(tile):
                num_nomask +=1
                continue

            name = f"tile_{event.name}_{x_idx}_{y_idx}.tif"
            save_path = Path(tile_path, name)
            #print('---save_path= ', save_path)

            if save_path.exists():
                save_path.unlink()

            # Save the multi-band GeoTIFF with layer names in metadata

            #print(f"---Saving tile {name}")
            tile.rio.to_raster(save_path, compress="deflate")
            num_saved += 1

    return total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask

def main():
    print('>>>>>in main')
    datacube_name = 'datacube_Vietnam_11alex.nc'
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326\Vietnam_11")
    datacube = rxr.open_rasterio(Path(data_root,datacube_name ))
    print(f"################### tiling ###################")
    event = Path(data_root)

    # DO THE TILING AND GET THE STATISTICS
    total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask = tile_datacube(datacube, event, tile_size=256, stride=256)

    print(f'>>>>total num of tiles: {total_num_tiles}')
    print(f'>>>>num of saved tiles: {num_saved}')
    print(f'>>>num witno valid data/analysis extent layer: {num_has_nans}')
    print(f'>>>num with no mask : {num_nomask}')
    print(f'>>>num with no valid layer : {num_novalid}')

    # check layers in tile
    tile_dir = Path(event, 'tiles') 
    # open a tile
    tile = rxr.open_rasterio(tile_dir / 'tile_Vietnam_11_0_0.tif')
    #print('>>>2 final saved tile= ', tile)
    print('---saved datacube crs = ', tile.rio.crs)

    
if __name__ == '__main__':
    main()

