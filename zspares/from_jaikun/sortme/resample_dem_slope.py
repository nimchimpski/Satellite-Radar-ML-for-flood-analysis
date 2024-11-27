#!/usr/bin/env python3
"""
STAC Data Processing Script

This Python script processes Sentinel-2, Sentinel-1, and Copernicus DEM
(Digital Elevation Model) data. It utilizes Microsoft's Planetary Computer API
for data retrieval and manipulation.

Constants:
- STAC_API: Planetary Computer API endpoint
- S2_BANDS: Bands used in Sentinel-2 data processing

Functions:
- get_surrounding_days(reference, interval_days):
      Get the week range for a given date.
- search_sentinel2(date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage):
      Search for Sentinel-2 items within a given date range and area of interest.
- search_sentinel1(aoi_bound, catalog, date_range):
      Search for Sentinel-1 items within a given bounding box, STAC catalog,
      and date range.
- search_dem(aoi_bound, catalog):
      Search for DEM items within a given bounding box.
- make_datasets(s2_item, s1_items, dem_items, resolution):
      Create xarray Datasets for Sentinel-2, Sentinel-1, and DEM data.
- process(aoi, year, resolution, cloud_cover_percentage, nodata_pixel_percentage):
      Process Sentinel-2, Sentinel-1, and DEM data for a specified time range,
      area of interest, and resolution.
"""

import logging
import os
import random
import glob
# import fiona
from datetime import timedelta, datetime

# import earthpy.spatial as es
# import earthpy.plot as ep
# import earthpy.mask as em

from osgeo import gdal

import click
import rasterio
import rioxarray # very important!
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from pystac import ItemCollection
from shapely.geometry import box
from shapely import Polygon, MultiPolygon, multipolygons
# from tile import tiler
import planetary_computer
# from fastkml import kml
from lxml import etree, objectify

# import osgeo.gdal as gdal
from rasterio.enums import ColorInterp
from rasterio.merge import merge
from shapely.geometry import mapping

from tqdm import tqdm
import pyproj
import json

planetary_computer.settings.set_subscription_key('0a584f3348c14138a912b7e443e2435e')

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
SPATIAL_RESOLUTION = 10
CLOUD_COVER_PERCENTAGE = 50
NODATA_PIXEL_PERCENTAGE = 20
NODATA = np.nan
S1_MATCH_ATTEMPTS = 40
DATES_PER_LOCATION = 4
TILE_SIZE = 256
VERSION=1
# NEW_DATASET= r'.\flood_uno_updated_v2'


logger = logging.getLogger("datacube")
hdr = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
hdr.setFormatter(formatter)
logger.addHandler(hdr)
logger.setLevel(logging.INFO)


def get_surrounding_days(reference, interval_days):
    """
    Get the days surrounding the input date.

    Parameters:
    - reference (datetime): The reference datetime.
    - interval_days (int): The number of days to search ahead and back

    Returns:
    - str: A string representing the start and end dates of the date interval in the
        format 'start_date/end_date'.
    """
    start = reference - timedelta(days=interval_days)
    end = reference + timedelta(days=interval_days)
    return f"{start.date()}/{end.date()}"


def search_sentinel1(polygon, catalog, date_range):
    """
    Search for Sentinel-1 items within a given bounding box (aoi_bound), STAC
    catalog, and date range.

    Parameters:
    - aoi_bound (tuple): Bounding box coordinates in the format
        (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing Sentinel-1 items.
    - date_range (str): The date range in the format 'start_date/end_date'.

    Returns:
    - pystac.Collection: A collection of Sentinel-1 items filtered by specified
        conditions.

    Note:
    This function retrieves Sentinel-1 items from the catalog that intersect
    with the given bounding box and fall within the provided time window. The
    function filters items based on orbit state and returns the collection of
    Sentinel-1 items that meet the defined criteria.
    """

    # Create a Polygon from the bounding box coordinates to save the time for searching
    aoi_bound = polygon.bounds
    aoi_bound_polygon = Polygon([
        (aoi_bound[0], aoi_bound[1]),  # min_x, min_y
        (aoi_bound[0], aoi_bound[3]),  # min_x, max_y
        (aoi_bound[2], aoi_bound[3]),  # max_x, max_y
        (aoi_bound[2], aoi_bound[1])   # max_x, min_y
    ])

    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, aoi_bound_polygon.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, date_range]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-1-rtc"]},
            ],
        },
    )
    s1_items = search.item_collection()
    logger.info(f"Found {len(s1_items)} Sentinel-1 items")

    return s1_items


def search_dem(bbox, catalog):
    """
    Search for Copernicus Digital Elevation Model (DEM) items within a given
    bounding box (aoi_bound), STAC catalog, and Sentinel-2 items.

    Parameters:
    - aoi_bound (tuple): Bounding box coordinates in the format
        (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing DEM items.

    Returns:
    - pystac.Collection: A collection of Digital Elevation Model (DEM) items
        filtered by specified conditions.
    """
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    dem_items = search.item_collection()
    logger.info(f"Found {len(dem_items)} DEM items")

    return dem_items


def make_datasets(s1_items, dem_items, reference):
    """
    Create xarray Datasets for Sentinel-1, and Copernicus DEM
    data.

    Parameters:
    - s1_items (list): List of Sentinel-1 items.
    - dem_items (list): List of DEM items.
    - resolution (int): Spatial resolution.

    Returns:
    - tuple: A tuple containing xarray Datasets for Sentinel-2, Sentinel-1,
        and Copernicus DEM.
    """

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items,
        assets=["vh", "vv"],
        epsg=4326,
        resolution=reference.res,
        bounds_latlon=reference.bounds,
        dtype=np.float32,
        fill_value=np.nan,
        snap_bounds=False
    )

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=int(da_sen1.epsg),
        resolution=0.0002777777777777778,
        bounds_latlon=reference.bounds,
        dtype=np.float32,
        fill_value=np.nan,
        snap_bounds=False,
    )

    da_sen1: xr.DataArray = stackstac.mosaic(da_sen1, dim="time")

    da_sen1 = da_sen1.drop_vars(
        [var for var in da_sen1.coords if var not in da_sen1.dims]
    )

    da_dem: xr.DataArray = stackstac.mosaic(da_dem, dim="time").assign_coords(
        {"band": ["dem"]}
    )

    da_dem = da_dem.drop_vars([var for var in da_dem.coords if var not in da_dem.dims])

    pixels = [da_sen1]

    return pixels, [da_dem]


def make_datasets_only_dem(dem_items, reference):
    """
    Create xarray Datasets for Sentinel-1, and Copernicus DEM
    data.

    Parameters:
    - s1_items (list): List of Sentinel-1 items.
    - dem_items (list): List of DEM items.
    - resolution (int): Spatial resolution.

    Returns:
    - tuple: A tuple containing xarray Datasets for Sentinel-2, Sentinel-1,
        and Copernicus DEM.
    """

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=4326,
        resolution=0.0002777777777777778,
        bounds_latlon=reference.bounds,
        dtype=np.float32,
        fill_value=np.nan,
        snap_bounds=False,
    )

    da_dem: xr.DataArray = stackstac.mosaic(da_dem, dim="time").assign_coords(
        {"band": ["dem"]}
    )

    da_dem = da_dem.drop_vars([var for var in da_dem.coords if var not in da_dem.dims])

    pixels = [da_dem.compute()]

    return pixels




def get_folders(path):
    folders = [l for l in os.listdir(path) \
               if os.path.isdir(os.path.join(path, l))]
    return folders




def make_datacube(location, save_path, catalog):
    date = location.split('\\')[-1].split('_')[1]
    start_date = datetime.strptime(date, '%Y%m%d')
    end_date = start_date + timedelta(days=1)
    daterange = '{}/{}'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))

    mask_path = glob.glob(os.path.join(location, "*_mask.tif"))[0]
    mask_img = rasterio.open(mask_path)
    
    # grd_path = glob.glob(os.path.join(location, "*compressed.tif"))[0]
    # grd_img = rasterio.open(grd_path)

    # analysis_extent_path = glob.glob(os.path.join(location, "*AnalysisExtent*.tif"))[0]
    # analysis_extent = rasterio.open(analysis_extent_path)

    s1_items = search_sentinel1(mask_img, catalog, daterange)
    dem_items = search_dem(mask_img.bounds, catalog)


    if len(s1_items) > 0 and len(dem_items) > 0:
        pixels, pixels_ds = make_datasets(s1_items, dem_items, mask_img)

        # da_mask = xr.DataArray([mask_img.read(1)], dims=['band', 'y', 'x'], coords={'band':['mask'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
        # da_mask.attrs = pixels[0].attrs
        # pixels.append(da_mask)

        # da_grd = xr.DataArray(grd_img.read(), dims=['band', 'y', 'x'], coords={'band':['grd'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
        # da_grd.attrs = pixels[0].attrs
        # pixels.append(da_grd)

        # da_analysis = xr.DataArray([analysis_extent.read(1)], dims=['band', 'y', 'x'], coords={'band':['analysis_extent'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
        # da_analysis.attrs = pixels[0].attrs
        # pixels.append(da_analysis)

        # for i, part in enumerate(pixels):
        #     if i == 0:
        #         part.rio.to_raster(os.path.join(save_path, 'sentinel-1-rtc.tif'), compress="deflate", bigtiff='Yes')
        #     elif i == 1:
        #         part.rio.to_raster(os.path.join(save_path, 'mask.tif'), compress="deflate", bigtiff='Yes')
        #     elif i == 2:
        #         part.rio.to_raster(os.path.join(save_path, 'sentinel-1-grd.tif'), compress="deflate", bigtiff='Yes')
        #     elif i == 3:
        #         part.rio.to_raster(os.path.join(save_path, 'analysis_extent.tif'), compress="deflate", bigtiff='Yes')

        for i, part in enumerate(pixels_ds):
            if i == 0:
                part.rio.to_raster(os.path.join(save_path, 'dem_updated.tif'), compress="deflate", bigtiff='Yes')

                dem_ds = gdal.Open(os.path.join(save_path, 'dem_updated.tif'))
                tmp_file = os.path.join(save_path, 'slope_updated.tif')
                gdal.DEMProcessing(tmp_file, dem_ds, 'slope', scale=111120)


    # elif len(s1_items) == 0:
        
    #     pixels = make_datasets_only_dem(dem_items, mask_img)

    #     da_mask = xr.DataArray([mask_img.read(1)], dims=['band', 'y', 'x'], coords={'band':['mask'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    #     da_mask.attrs = pixels[0].attrs
    #     pixels.append(da_mask)

    #     da_grd = xr.DataArray(grd_img.read(), dims=['band', 'y', 'x'], coords={'band':['grd'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    #     da_grd.attrs = pixels[0].attrs
    #     pixels.append(da_grd)

    #     da_analysis = xr.DataArray([analysis_extent.read(1)], dims=['band', 'y', 'x'], coords={'band':['analysis_extent'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    #     da_analysis.attrs = pixels[0].attrs
    #     pixels.append(da_analysis)

    #     for i, part in enumerate(pixels):
    #         if i == 0:
    #             part.rio.to_raster(os.path.join(save_path, 'dem.tif'), compress="deflate", bigtiff='Yes')
    #         elif i == 1:
    #             part.rio.to_raster(os.path.join(save_path, 'slope.tif'), compress="deflate", bigtiff='Yes')
    #         elif i == 2:
    #             part.rio.to_raster(os.path.join(save_path, 'mask.tif'), compress="deflate", bigtiff='Yes')
    #         elif i == 3:
    #             part.rio.to_raster(os.path.join(save_path, 'sentinel-1-grd.tif'), compress="deflate", bigtiff='Yes')
    #         elif i == 4:
    #             part.rio.to_raster(os.path.join(save_path, 'analysis_extent.tif'), compress="deflate", bigtiff='Yes')



def handle_event(root, event, catalog):
    current_path = os.path.join(root, event)

    # os.makedirs(os.path.join(NEW_DATASET, event), exist_ok=True)

    logger.info("handling [{}]".format(event))

    files = [f.split('\\')[-1] for f in glob.glob(os.path.join(NEW_DATASET, event, '*.tif'))]
    if 'dem_updated.tif' in files:
        print("Continue to next event...")
    else:
        make_datacube(current_path, os.path.join(NEW_DATASET, event), catalog)


NEW_DATASET = r'.\flood_uno_updated_v4'

@click.command()
@click.option('--event', '-e', type=str)
def main(event):
    old_flooduno_root = r"Y:\Users\Jiakun\FloodAI\scripts\flood-55\scripts\flood_uno_updated_v4"
    flood_events = get_folders(old_flooduno_root)

    print("There are {} events in this folder".format(len(flood_events)))

    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

    for event in tqdm(flood_events):
        handle_event(old_flooduno_root, event, catalog)


if __name__ == "__main__":
    main()