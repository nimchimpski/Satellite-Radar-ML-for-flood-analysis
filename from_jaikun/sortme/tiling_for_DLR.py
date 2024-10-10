# import logging
import os
# import random
# import glob
# import fiona
# from datetime import timedelta, datetime
# import math
# import earthpy.spatial as es
# import earthpy.plot as ep
# import earthpy.mask as em
# from osgeo import gdal, ogr
# import tifffile as tiff
# import click
import rasterio
import rioxarray # very important!
# import geopandas as gpd
import numpy as np
# import planetary_computer as pc
# import pystac_client
# import stackstac
import xarray as xr
# from pystac import ItemCollection
# from shapely.geometry import box
# from shapely import Polygon, MultiPolygon, multipolygons
# from tile import tiler
# import planetary_computer
# from fastkml import kml
# from lxml import etree, objectify
# import osgeo.gdal as gdal
from rasterio.enums import ColorInterp
# from rasterio.merge import merge
# from shapely.geometry import mapping
from tqdm import tqdm
# import pyproj
# import json

VERSION= 'DLR-water'

def filter_nodata(tile):
    """
    Filter tiles based on no-data pixels.
    Args:
    - tile (xarray.Dataset): A subset of data representing a tile.
    Returns:
    - bool: True if the tile is approved, False if rejected.
    """
    # bands_to_check = ["dem", "slope"]
    # for band in bands_to_check:
    #     if int(np.isnan(tile.sel(band=band)).sum()):
    #         # print(f"Too much no-data in {band}")
    #         return False
    
    if int(np.isnan(tile.sel(band='vv')).sum()):
        return False
    
    if int(np.isnan(tile.sel(band='vh')).sum()):
        return False
        
    return True  # If both conditions pass

def filter_nomask(tile):
        # if a mask is has no usefull information, we skip the tile. this is because of class imbalance
    if (tile.sel(band='mask').sum().values.tolist() == 0):
        return False        
    return True  # If both conditions pass

def filter_noanalysis(tile):
    if 0 in tile.sel(band='analysis_extent').values:
        return False
    
    if int(np.isnan(tile.sel(band='analysis_extent')).sum()):
        return False

    return True


def tile_to_dir(stack, path, tile_size, stride, event):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    - bucket(str): AWS S3 bucket to write tiles to
    """
    print(f"Writing tempfiles to {path}")
    os.makedirs(path, exist_ok=True)

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
            parts = [part['band_data'][:, y_start:y_end, x_start:x_end] for part in stack]

            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 250 == 0:
                print("Counted {} tiles".format(counter))

            # set the criterion here.
            if not filter_nodata(tile):
                continue

            if not filter_noanalysis(tile):
                continue

            if not filter_nomask(tile):
                counter_nomask +=1
                continue

            # if (tile.sel(band='mask').sum().values.tolist() == 0):
            #     import pdb; pdb.set_trace()

            # Track band names and color interpretation
            tile.attrs["long_name"] = [str(x.values) for x in tile.band]
            color = [ColorInterp.blue, ColorInterp.green, ColorInterp.red] + [
                ColorInterp.gray
            ] * (len(tile.band) - 3)

            name = os.path.join(path, "tile_{event}_{version}_{x_idx}_{y_idx}.tif".format(
                path=path,
                event=event,
                version=VERSION,
                x_idx=x_idx,
                y_idx=y_idx
            ))
            tile.rio.to_raster(name, compress="deflate", nodata=np.nan)
            counter_valid += 1

            with rasterio.open(name, "r+") as rst:
                rst.colorinterp = color
                # rst.update_tags(date=VERSION)

    return counter, counter_valid, counter_nomask


if __name__ == "__main__":
    data_root = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\DLR_S1S2\S1S2_water'

    events = os.listdir(data_root)
    norm = True
    tile_statistics = {}
    for event in tqdm(events):

        pixels = []

        if os.path.exists(os.path.join(data_root, event, 'tiles_norm')) and len(os.listdir(os.path.join(data_root, event, 'tiles'))) > 0:
            continue

        datas = {
            'sentinel12_s1_{}_img-original.tif'.format(event): ['vv', 'vh'],
            'sentinel12_s1_{}_msk.tif'.format(event): ['mask'],
            'sentinel12_s1_{}_valid.tif'.format(event): ['analysis_extent'],
            'sentinel12_copdem30_{}_elevation.tif'.format(event): ['dem'],
            'sentinel12_copdem30_{}_slope.tif'.format(event): ['slope'],
        }

        da_s1 = xr.open_dataset(os.path.join(data_root, event, f'sentinel12_s1_{event}_img-original.tif'))
        # if norm:
        #     da_s1_array = da_s1.to_array().to_numpy()[0]
        #     da_s1_array[da_s1_array==0] = np.nan
        #     min_val = np.nanpercentile(da_s1_array[0], 1)
        #     max_val = np.nanpercentile(da_s1_array[0], 99)
        #     vv_norm = ((da_s1_array[0] - min_val) / (max_val - min_val)).clip(0,1)
        da_s1 = da_s1.assign_coords({'band': ['vv', 'vh']})

        da_msk = xr.open_dataset(os.path.join(data_root, event, f'sentinel12_s1_{event}_msk.tif'))
        da_msk = da_msk.assign_coords({'band': ['mask']})

        da_valid = xr.open_dataset(os.path.join(data_root, event, f'sentinel12_s1_{event}_valid.tif'))
        da_valid = da_valid.assign_coords({'band': ['analysis_extent']})

        da_dem = xr.open_dataset(os.path.join(data_root, event, f'sentinel12_copdem30_{event}_elevation.tif'))
        # new_x = np.linspace(da_s1.x.min(), da_s1.x.max(), da_s1.x.sizes.get('x'))
        # new_y = np.linspace(da_s1.y.min(), da_s1.y.max(), da_s1.y.sizes.get('y'))
        da_dem = da_dem.reindex({'x': da_s1.x, 'y':da_s1.y}, method='nearest')
        da_dem = da_dem.assign_coords({'band': ['dem']})

        da_slope = xr.open_dataset(os.path.join(data_root, event, f'sentinel12_copdem30_{event}_slope.tif'))
        # new_x = np.linspace(da_s1.x.min(), da_s1.x.max(), da_s1.x.sizes.get('x'))
        # new_y = np.linspace(da_s1.y.min(), da_s1.y.max(), da_s1.y.sizes.get('y'))
        da_slope = da_slope.reindex({'x': da_s1.x, 'y':da_s1.y}, method='nearest')
        da_slope = da_slope.assign_coords({'band': ['slope']})

        pixels = [da_s1, da_dem, da_slope, da_msk, da_valid]

        num_tiles, num_valid_tiles, num_nomask_tiles = tile_to_dir(pixels, os.path.join(data_root, event, 'tiles'), tile_size=256, stride=256, event=event)
        tile_statistics[event] = [num_tiles, num_valid_tiles, num_nomask_tiles]
