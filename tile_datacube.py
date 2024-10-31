import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import ColorInterp
import rioxarray as rxr 
from pathlib import Path

def tile_datacube(datacube, event, tile_size=256, stride=256):
    """
    Tile a datacube and save to 'tiles' dir in same location.
    'ARG event is a Path object
    play with the stride and tile_size to get the right size of tiles.

    """
    print('\n+++++++in tile_to_dir fn++++++++')

    def contains_nodata(tile):
        """
        Checks if specified bands contain NaN values.
        Returns True if NaNs are found, along with the count of NaNs per band.
        Returns False if no NaNs are found.
        """
        bands_to_check = ["dem", "slope", "vv", "vh"]
        contains_nan = False  # Flag to track if any NaNs are found

        sub_array = tile['__xarray_dataarray_variable__']

        for band in bands_to_check:
            # Calculate the number of NaNs in the current band
            nan_count = np.isnan(sub_array.sel(layer=band)).sum().values
            if nan_count > 0:
                print(f"Tile contains {nan_count} NaN values in {band}")
                contains_nan = True

        return contains_nan  # True if any NaNs were found, False if none

    def has_no_valid(tile):
        '''
        Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
        return False = ok
        '''
        sub_array = tile['__xarray_dataarray_variable__']

        # Ensure 'valid' layer exists
        if 'valid' not in sub_array.coords['layer'].values:
            print("Warning: 'valid' layer not found in tile.")
            return True  # If 'valid' layer is missing, treat as no valid data

        return 1 not in sub_array.sel(layer='valid').values
    
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
    
    print('---loaded datacube= ', datacube)
    print('==============================')
    # num_x_tiles = max(datacube.x.size + stride - 1, 0) // stride + 1
    # num_y_tiles = max(datacube.y.size + stride - 1, 0) // stride + 1

    num_x_tiles = 1

    num_y_tiles = 1


    counter = 0
    counter_valid = 0
    counter_nomask = 0
    for y_idx in range(num_y_tiles):

        for x_idx in range(num_x_tiles):

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, datacube.x.size)
            y_end = min(y_start + tile_size, datacube.y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            tile = datacube.sel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            print('--- tile as dataset= ', tile)
            print('===================================================')

            
            counter += 1
            num_novalid = 0
            num_nomask = 0
            num_nodata = 0
            if counter % 250 == 0:
                print(f"Counted {counter} tiles")

            # set the criterion here.
            if contains_nodata(tile):
                num_nodata += 1
                continue

            if has_no_valid(tile):
                num_novalid += 1
                continue

            if has_no_mask(tile):
                num_nomask +=1
                continue

            name = f"tile_{event.name}_{x_idx}_{y_idx}.tif"
            save_path = Path(tile_path, name)
            print('---save_path= ', save_path)

            # rio.to_raster needs to work on a numpy array, so we need to convert the xarray to a numpy array
            tile_array = tile['__xarray_dataarray_variable__']
            print('---tile as array= ', tile_array)
            # layer_names = ["dem", "slope", "mask", "valid", "vh", "vv"]
            # tile_array.assign_coords["band"] = layer_names
            layer_names = ["dem", "slope", "mask", "valid", "vh", "vv"]
            # Remove existing file if needed
            if save_path.exists():
                save_path.unlink()

            # Save the multi-band GeoTIFF with layer names in metadata
            print(f"---Saving tile {name}")

            with rasterio.open(
                save_path,
                "w",
                driver="GTiff",
                height=tile_array.shape[1],
                width=tile_array.shape[2],
                count=tile_array.shape[0],
                dtype=tile_array.dtype,
                compress="deflate"
            ) as dst:
                # Write each band and set its description
                for i in range(tile_array.shape[0]):
                    dst.write(tile_array[i], i + 1)  # Write band data (1-based index)
                    dst.set_band_description(i + 1, layer_names[i])  # Set band description (name)

                # Add layer names as a custom metadata tag
                dst.update_tags(layer_names=",".join(layer_names))

            print(f"Saved GeoTIFF with embedded layer names at {save_path}")


            counter_valid += 1
            # break

        # break

    return counter, counter_valid, counter_nomask

def main():
    print('>>>>>in main')
    datacube_name = 'datacube_Vietnam_11alex.nc'
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326\Vietnam_11")
    datacube = rxr.open_rasterio(Path(data_root,datacube_name ))
    print(f"################### tiling ###################")
    event = Path(data_root)
    # DO THE TILING AND GET THE STATISTICS
    counter, num_novalid, num_nomask = tile_datacube(datacube, event, tile_size=256, stride=256)

    print('>>>>num of tiles:', counter)
    print(f'>>>>num of valid tiles: {counter - num_novalid}')
    print(f'>>>num of tiles without mask: {num_nomask}')

    # check layers in tile
    tile_dir = Path(event, 'tiles') 
    # open a tile
    tile = rxr.open_rasterio(tile_dir / 'tile_Vietnam_11_0_0.tif')
    print('>>>final tile= ', tile)
    
if __name__ == '__main__':
    main()

