import os
import rasterio
import rioxarray # very important!
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import xarray as xr

from rasterio.enums import ColorInterp
from scipy.ndimage import map_coordinates

from tqdm import tqdm
import cv2

from sklearn import preprocessing as pre

VERSION = "UNOSAT-AI-v2"
TRACKING = "UNOSAT-AI-v1"


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
            # print(f"Too much no-data in {band}")
            return False
        
    if 0 in tile.sel(band='grd').values:
        return False
        
    return True  # If both conditions pass

def filter_nomask(tile):
    if (tile.sel(band='mask').sum().values.tolist() == 0):
        return False        
    return True  # If both conditions pass


def tile_to_dir(stack, date, dir, tile_size, stride, country):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    - bucket(str): AWS S3 bucket to write tiles to
    """
    print(f"Writing tempfiles to {dir}")
    os.makedirs(dir, exist_ok=True)

    # Calculate the number of full tiles in x and y directions
    # num_x_tiles = stack[0].x.size // tile_size
    # num_y_tiles = stack[0].y.size // tile_size
    num_x_tiles = max(stack[0].x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(stack[0].y.size + stride - 1, 0) // stride + 1

    counter = 0
    counter_valid = 0
    counter_nomask = 0
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            # Calculate the start and end indices for x and y dimensions
            # for the current tile
            # x_start = x_idx * tile_size
            # y_start = y_idx * tile_size
            # x_end = min(stride * x_idx + tile_size, stack[0].x.size)
            # y_end = min(stride * y_idx + tile_size, stack[0].y.size)

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, stack[0].x.size)
            y_end = min(y_start + tile_size, stack[0].y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            parts = [part[:, y_start:y_end, x_start:x_end] for part in stack]

            # Only concat here to save memory, it converts S2 data to float
            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 250 == 0:
                print("Counted {} tiles".format(counter))


            # set the criterion here.
            if not filter_nodata(tile):
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
    data_root = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\flood_uno_updated_v2'

    events = os.listdir(data_root)

    datas = {
        'sentinel-1-grd.tif': 'grd',
        'dem_updated.tif': 'dem',
        'slope_updated.tif': 'slope',
        'mask.tif': 'mask'
        }
    
    tile_statistics = {}

    for event in tqdm(events):
        if event == 'ST1_20180501_SOM':
            continue
        date = event.split('_')[1]
        country = event.split('_')[-1]
        pixels = []

        if 'tiles_norm_v1' in os.listdir(os.path.join(data_root, event)):
            continue

        def check_rtc_band():
            da = xr.open_dataset(os.path.join(data_root, event, 'sentinel-1-rtc.tif'), engine='rasterio')
            print(da.dims['band'])

        da_rtc_origin = xr.open_dataset(os.path.join(data_root, event, 'sentinel-1-rtc.tif'), engine='rasterio')
        band_size, x_size, y_size = da_rtc_origin.sizes['band'], da_rtc_origin.sizes['x'], da_rtc_origin.sizes['y']
        if band_size == 1:
            rtc_array = da_rtc_origin.to_array().to_numpy()[0,0]
            min_val = np.nanpercentile(rtc_array, 2)
            max_val = np.nanpercentile(rtc_array, 98)
            rtc_norm = ((rtc_array - min_val) / (max_val - min_val)).clip(0,1)
            rtc_norm_new = ((rtc_array - min_val) / (max_val - min_val)).clip(0,1)
            da_rtc = xr.DataArray([rtc_norm, rtc_norm], dims=['band', 'y', 'x'], coords={'band':['vv', 'vh'], 'y': da_rtc_origin['y'], 'x':da_rtc_origin['x']})
            da_rtc.attrs = da_rtc_origin.attrs
            pixels.append(da_rtc)
        else:
            rtc_array = da_rtc_origin.to_array().to_numpy()[0]
            min_val = np.nanpercentile(rtc_array[0], 2)
            max_val = np.nanpercentile(rtc_array[0], 98)
            rtc_norm0 = ((rtc_array[0] - min_val) / (max_val - min_val)).clip(0,1)
            min_val = np.nanpercentile(rtc_array[1], 2)
            max_val = np.nanpercentile(rtc_array[1], 98)
            rtc_norm1 = ((rtc_array[1] - min_val) / (max_val - min_val)).clip(0,1)
            da_rtc = xr.DataArray([rtc_norm0, rtc_norm1], dims=['band', 'y', 'x'], coords={'band':['vv', 'vh'], 'y': da_rtc_origin['y'], 'x':da_rtc_origin['x']})
            pixels.append(da_rtc)

        for k, v in datas.items():
            print("loading {}".format(k))
            da = xr.open_dataset(os.path.join(data_root, event, k), engine='rasterio')
            da = da.assign_coords({'band':[v]})
            if k == 'dem_updated.tif' or k == 'slope_updated.tif':
                da_array = da.to_array().to_numpy()
                resized = cv2.resize(da_array[0,0], (x_size, y_size))
                da = xr.DataArray([resized], dims=['band', 'y', 'x'], coords={'band':[v], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
            elif k == 'sentinel-1-grd.tif':
                da_array = da.to_array().to_numpy()
                da_norm = pre.MinMaxScaler().fit_transform(da_array[0,0])
                da = xr.DataArray([da_norm], dims=['band', 'y', 'x'], coords={'band':[v], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
            else:
                da_array = da.to_array().to_numpy()
                da = xr.DataArray([da_array[0,0]], dims=['band', 'y', 'x'], coords={'band':[v], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
            da.attrs = da_rtc_origin.attrs
            pixels.append(da)

        # VV, VH, GRD, DEM, SLOPE, MASK

        num_tiles, num_valid_tiles, num_nomask_tiles = tile_to_dir(pixels, date, os.path.join(data_root, event, 'tiles_norm'), tile_size=256, stride=256, country=country)
        tile_statistics[event] = [num_tiles, num_valid_tiles, num_nomask_tiles]

