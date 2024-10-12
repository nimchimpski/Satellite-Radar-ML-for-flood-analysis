import os
import rasterio
import rioxarray # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

VERSION = "UNOSAT-AI-v2"
TRACKING = "UNOSAT-AI-v1"


def custom_normalize(array, lower_percentile=2, upper_percentile=98, clip_range=(0, 1)):
    """
    Normalizes a given array by scaling values between the given percentiles
    and clipping to the specified range.

    Args:
    - array: Input data (NumPy array or xarray DataArray)
    - lower_percentile: The lower bound for normalization (default 2)
    - upper_percentile: The upper bound for normalization (default 98)
    - clip_range: Tuple specifying the range to clip the normalized data (default (0, 1))

    Returns:
    - normalized_array: The normalized data array
    """
    # Compute percentiles
    min_val = np.nanpercentile(array, lower_percentile)
    max_val = np.nanpercentile(array, upper_percentile)
    
    # Normalize the array
    normalized_array = (array - min_val) / (max_val - min_val)
    
    # Clip values to the specified range
    normalized_array = np.clip(normalized_array, clip_range[0], clip_range[1])
    
    return normalized_array


def filter_nodata(tile):
    """
    Filter tiles based on no-data pixels.
    Args:
    - tile (xarray.Dataset): A subset of data representing a tile.
    Returns:
    - bool: True if the tile is approved, False if rejected.
    """
    bands_to_check = ["dem", "slope", "vv", "vh"]
    for band in bands_to_check:
        if int(np.isnan(tile.sel(band=band)).sum()):
            return False
    return True  # If both conditions pass

def filter_nomask(tile):
    if (tile.sel(band='mask').sum().values.tolist() == 0):
        return False        
    return True  # If both conditions pass

def filter_noanalysis(tile):
    '''
    Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
    '''
    if 0 in tile.sel(band='analysis_extent').values:
        return False
    return True


def tile_to_dir(stack, date, dir, tile_size, stride, country):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.
    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    """
    print(f"Writing tempfiles to {dir}")
    os.makedirs(dir, exist_ok=True)

    num_x_tiles = max(stack[0].x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(stack[0].y.size + stride - 1, 0) // stride + 1

    counter = 0
    counter_valid = 0
    counter_nomask = 0
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, stack[0].x.size)
            y_end = min(y_start + tile_size, stack[0].y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            parts = [part[:, y_start:y_end, x_start:x_end] for part in stack]
            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 250 == 0:
                print("Counted {} tiles".format(counter))

            # Filtering tiles based on various criteria
            if not filter_nodata(tile):
                continue

            if not filter_noanalysis(tile):
                continue

            if not filter_nomask(tile):
                counter_nomask +=1 
                continue

            # Track band names and color interpretation
            tile.attrs["long_name"] = [str(x.values) for x in tile.band]
            color = [ColorInterp.blue, ColorInterp.green, ColorInterp.red] + [
                ColorInterp.gray
            ] * (len(tile.band) - 3)

            name = os.path.join(dir, "tile_{date}_{country}_{version}_{x_idx}_{y_idx}.tif".format(
                dir=dir,
                date=date.replace("-", ""),
                country=country,
                version=VERSION,
                x_idx=x_idx,
                y_idx=y_idx
            ))
            tile.rio.to_raster(name, compress="deflate", nodata=np.nan)
            counter_valid += 1

            tags = {"version": "v2", "tracking": TRACKING}

            with rasterio.open(name, "r+") as rst:
                rst.colorinterp = color
                rst.update_tags(date=date)
                rst.update_tags(ns='tracking', tracking=TRACKING)

    return counter, counter_valid, counter_nomask


if __name__ == "__main__":
    data_root = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\flood_uno_updated_v2_HITL'

    events = os.listdir(data_root)

    datas = {
        'sentinel-1-img.tif': ['vv', 'vh'],
        'dem_updated.tif': 'dem',
        'slope_updated.tif': 'slope',
        'mask.tif': 'mask',
        'valid.tif': 'analysis_extent'
    }
    
    tile_statistics = {}

    for event in tqdm(events):
        date = event.split('_')[1]
        country = event.split('_')[-1]
        pixels = []

        if 'tiles_norm' in os.listdir(os.path.join(data_root, event)):
            continue

        # Loop through each dataset to normalize and load into xarray
        for k, v in datas.items():
            print("loading {}".format(k))
            da = xr.open_dataset(os.path.join(data_root, event, k), engine='rasterio')
            
            if k == 'sentinel-1-img.tif':
                da = da.assign_coords({'band': v})  # Assign 'vv' and 'vh'
                da_array = da.to_array().to_numpy()

                # Normalizing both VV and VH bands
                vv_norm = custom_normalize(da_array[0])
                vh_norm = custom_normalize(da_array[1])

                da = xr.DataArray([vv_norm, vh_norm], dims=['band', 'y', 'x'], coords={'band': v, 'y': da.y, 'x': da.x})
            else:
                da_array = da.to_array().to_numpy()

                # Normalize other data (dem, slope, mask, analysis_extent)
                norm_array = custom_normalize(da_array[0])
                da = xr.DataArray([norm_array], dims=['band', 'y', 'x'], coords={'band': [v], 'y': da.y, 'x': da.x})
            
            pixels.append(da)

        # Now tile the dataset after normalization
        num_tiles, num_valid_tiles, num_nomask_tiles = tile_to_dir(pixels, date, os.path.join(data_root, event, 'tiles_norm'), tile_size=256, stride=256, country=country)
        tile_statistics[event] = [num_tiles, num_valid_tiles, num_nomask_tiles]
