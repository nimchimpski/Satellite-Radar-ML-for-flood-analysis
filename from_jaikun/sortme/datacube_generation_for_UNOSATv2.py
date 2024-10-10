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
import fiona
from datetime import timedelta, datetime

import earthpy.spatial as es
import earthpy.plot as ep
import earthpy.mask as em

from osgeo import gdal, ogr

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
from fastkml import kml
from lxml import etree, objectify

import osgeo.gdal as gdal
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
NEW_DATASET= r'.\flood_uno_updated_v4'

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


def make_datasets(s1_items, dem_items, resolution, reference, shapefile):
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

    # TODO: maybe we need to change the resolutoin according to the shapefile
    # from rasterio.crs import CRS
    # reprojected_bounds = rasterio.warp.transform_bounds(shapefile.crs, CRS.from_epsg(s1_items[0].properties['proj:epsg']), *shapefile.bounds)

    ref = reference[0]

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items,
        assets=["vh", "vv"],
        epsg=4326,
        resolution=ref.res,
        bounds_latlon=shapefile.bounds,
        dtype=np.float32,
        fill_value=np.nan,
        snap_bounds=False
    )

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=int(da_sen1.epsg),
        bounds_latlon=shapefile.bounds,
        resolution=ref.res,
        dtype=np.float32,
        fill_value=np.nan,
        snap_bounds=False
    )

    da_sen1: xr.DataArray = stackstac.mosaic(da_sen1, dim="time")

    da_sen1 = da_sen1.drop_vars(
        [var for var in da_sen1.coords if var not in da_sen1.dims]
    )

    da_dem: xr.DataArray = stackstac.mosaic(da_dem, dim="time").assign_coords(
        {"band": ["dem"]}
    )

    da_dem = da_dem.drop_vars([var for var in da_dem.coords if var not in da_dem.dims])

    da_slope = da_dem.copy(deep=True).assign_coords({"band": ["slope"]})

    pixels = [da_sen1, da_dem.compute()]


    # 1. Calculate slope from DEM with gdal.DEMProcessing
    # dem_ds = gdal.Open(r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\data\dem.tif')
    # tmp_file = r'Y:\Users\Jiakun\FloodAI\scripts\flood-55\data\slope_tmp.tif'
    # gdal.DEMProcessing(tmp_file, dem_ds, 'slope')
    # slope = gdal.Open(tmp_file).ReadAsArray()

    # 2. Calculate slope from DEM with gradient
    # https://gis.stackexchange.com/questions/361837/calculating-slope-of-numpy-array-using-gdal-demprocessing
    def gradient2degree(x):
        x_rad = np.arctan(np.sqrt(x))
        x_degree = np.rad2deg(x_rad)
        return x_degree

    da_slope = xr.apply_ufunc(gradient2degree, (da_slope.differentiate("x")**2 + da_slope.differentiate("y")**2).compute())
    da_slope.attrs = da_dem.attrs
    
    pixels.append(da_slope)


    return pixels


def filter_nodata(tile):
    """
    Filter tiles based on no-data pixels.

    Args:
    - tile (xarray.Dataset): A subset of data representing a tile.

    Returns:
    - bool: True if the tile is approved, False if rejected.
    """
    # Check for nodata pixels
    if 0 in tile.sel(band='analysis_extent').values:
        return False

    bands_to_check = ["vv", "vh", "dem", "slope"]
    for band in bands_to_check:
        if int(np.isnan(tile.sel(band=band)).sum()):
            logger.debug(f"Too much no-data in {band}")
            return False
        
    return True  # If both conditions pass


def tile_to_dir(stack, date, dir):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    - bucket(str): AWS S3 bucket to write tiles to
    """
    logger.debug(f"Writing tempfiles to {dir}")
    os.makedirs(dir, exist_ok=True)

    # Calculate the number of full tiles in x and y directions
    num_x_tiles = stack[0].x.size // TILE_SIZE
    num_y_tiles = stack[0].y.size // TILE_SIZE

    counter = 0
    # Iterate through each chunk of x and y dimensions and create tiles
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            # Calculate the start and end indices for x and y dimensions
            # for the current tile
            x_start = x_idx * TILE_SIZE
            y_start = y_idx * TILE_SIZE
            x_end = x_start + TILE_SIZE
            y_end = y_start + TILE_SIZE

            # Select the subset of data for the current tile
            parts = [part[:, y_start:y_end, x_start:x_end] for part in stack]

            # Only concat here to save memory, it converts S2 data to float
            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 250 == 0:
                logger.info(f"Counted {counter} tiles")


            # set the criterion here.
            if not filter_nodata(tile):
                continue


            # tile = tile.drop_sel(band="SCL")

            # Track band names and color interpretation
            tile.attrs["long_name"] = [str(x.values) for x in tile.band]
            color = [ColorInterp.blue, ColorInterp.green, ColorInterp.red] + [
                ColorInterp.gray
            ] * (len(tile.band) - 3)

            # Write tile to tempdir
            name = os.path.join(dir, "tile_{date}_v{version}_{counter}.tif".format(
                dir=dir,
                date=date.replace("-", ""),
                version=VERSION,
                counter=str(counter).zfill(8),
            ))
            tile.rio.to_raster(name, compress="deflate")

            with rasterio.open(name, "r+") as rst:
                rst.colorinterp = color
                rst.update_tags(date=date)

    return counter


def convert_attrs_and_coords_objects_to_str(data):
    """
    Convert attributes and coordinates that are objects to
    strings.

    This is required for storing the xarray in netcdf.
    """
    for key, coord in data.coords.items():
        if coord.dtype == "object":
            data.coords[key] = str(coord.values)

    for key, attr in data.attrs.items():
        data.attrs[key] = str(attr)

    for key, var in data.variables.items():
        var.attrs = {}

def convert_kml_to_polygon(KMLPath):
    route = objectify.fromstring(open(KMLPath).read().encode()) ##for python 3.6

    r = route.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text
    # r = route.Document.Folder.Placemark.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates.text
    r = r.split(' ')
    n = len(r)
    l = r[1:n-1]
    l.insert(0, r[n-2])
    coords = [[float(coord.split(',')[0]), float(coord.split(',')[1])] for coord in l]
    polygon = multipolygons([coords])

    return polygon

def get_folders(path):
    folders = [l for l in os.listdir(path) \
               if os.path.isdir(os.path.join(path, l))]
    return folders

def get_specific_folder(path, keyword=None):
    if keyword:
        folders = get_folders(path)
        for folder in folders:
            if keyword in folder:
                return os.path.join(path, folder)

    return False

def load_mask(path, mask_shp):
    mask_shapefile = gpd.read_file(mask_shp)
    feature = [mapping(mask_shapefile.geometry.values[0])]

    mask_files = glob.glob(os.path.join(path, '*_Orb_mask.tif'))
    if len(mask_files) == 0:
        mask_files = glob.glob(os.path.join(path, '*_mask.tif'))
    assert len(mask_files) > 0

    masks = []
    for mask_file in mask_files:
        mask = rasterio.open(os.path.join(path, mask_file))
        masks.append(mask)
        # cropped_mask, out_transform = rasterio.mask.mask(mask, feature, crop=True)
        # masks.append(cropped_mask)

    return masks


def make_datacube(location, save_path, catalog):
    im_path = os.path.join(location, 'image')
    im_reference = glob.glob(os.path.join(im_path, '*_Orb_Spk_TC_complete.tif'))[0]
    start_date = datetime.strptime(im_reference.split('\\')[-1].split('_')[4][:8], '%Y%m%d')
    end_date = start_date + timedelta(days=1)
    daterange = '{}/{}'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))

    gt_path = os.path.join(location, 'gt')
    mask_path = get_specific_folder(gt_path, 'ValidationMask')
    mask_shp = glob.glob(os.path.join(mask_path, '*.shp'))
    assert len(mask_shp) == 1
    mask_shp = mask_shp[0]
    mask_shapefile = fiona.open(mask_shp)

    mask_ims = glob.glob(os.path.join(mask_path, "*_mask.tif"))
    mask_imgs = [rasterio.open(imp) for imp in mask_ims]
    mask_final, mask_transform = merge(mask_imgs, bounds=mask_shapefile.bounds)

    s1_images = glob.glob(os.path.join(im_path, '*_compressed.tif'))
    s1_images = [rasterio.open(imp) for imp in s1_images]
    s1_final, s1_transform = merge(s1_images, bounds=mask_shapefile.bounds)


    analysis_path = get_specific_folder(gt_path, 'AnalysisExtent')
    analysis_files = glob.glob(os.path.join(analysis_path, '*_mask.tif'))
    analysis_images = [rasterio.open(file) for file in analysis_files]
    analysis_final, analysis_transform = merge(analysis_images, bounds=mask_shapefile.bounds)

    s1_items = search_sentinel1(mask_shapefile, catalog, daterange)
    dem_items = search_dem(mask_shapefile.bounds, catalog)

    # gather together
    pixels = make_datasets(s1_items, dem_items, SPATIAL_RESOLUTION, mask_imgs, mask_shapefile)

    # height, width = pixels[0].shape[2], pixels[0].shape[1]
    # bbox = mask_shapefile.bounds
    # transform = rasterio.transform.from_bounds(*bbox, height, width)
    # feature = [mapping(gpd.read_file(mask_shp).geometry.values[0])]
    # mask_final = rasterio.features.rasterize(feature,out_shape=(width, height),transform=transform,all_touched=True)


    da_mask = xr.DataArray(mask_final, dims=['band', 'y', 'x'], coords={'band':['mask'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    da_mask.attrs = pixels[0].attrs
    pixels.append(da_mask)

    da_s1_final = xr.DataArray(s1_final, dims=['band', 'y', 'x'], coords={'band':['vh'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    da_s1_final.attrs = pixels[0].attrs
    pixels.append(da_s1_final)

    da_analysis_extent = xr.DataArray(analysis_final, dims=['band', 'y', 'x'], coords={'band':['analysis_extent'], 'y': pixels[0]['y'], 'x':pixels[0]['x']})
    da_analysis_extent.attrs = pixels[0].attrs
    pixels.append(da_analysis_extent)

    for i, part in enumerate(pixels):
        if i == 0:
            part.rio.to_raster(os.path.join(save_path, 'sentinel-1-rtc.tif'), compress="deflate", bigtiff='Yes')
        elif i == 1:
            part.rio.to_raster(os.path.join(save_path, 'dem.tif'), compress="deflate", bigtiff='Yes')
        elif i == 2:
            part.rio.to_raster(os.path.join(save_path, 'slope.tif'), compress="deflate", bigtiff='Yes')
        elif i == 3:
            part.rio.to_raster(os.path.join(save_path, 'mask.tif'), compress="deflate", bigtiff='Yes')
        elif i == 4:
            part.rio.to_raster(os.path.join(save_path, 'sentinel-1-grd.tif'), compress="deflate", bigtiff='Yes')
        elif i == 5:
            part.rio.to_raster(os.path.join(save_path, 'analysis-extent.tif'), compress="deflate", bigtiff='Yes')

    # tiles_counter = tile_to_dir(pixels, str(s1_items[0].datetime.date()), os.path.join(save_path, 'tiles'))
    # return tiles_counter
    return 0


def handle_event(root, event, catalog):
    current_path = os.path.join(root, event)
    sub_locations = get_folders(current_path)

    os.makedirs(os.path.join(NEW_DATASET, event), exist_ok=True)

    counter_dic = {}

    for location in sub_locations:
        logger.info("handling [{}]".format(location))
        os.makedirs(os.path.join(NEW_DATASET, event, location), exist_ok=True)
        if len(os.listdir(os.path.join(NEW_DATASET, event, location))) > 0:
            print("Tiles already there, skip this event.")
            continue
        tiles_counter = make_datacube(os.path.join(current_path, location), os.path.join(NEW_DATASET, event, location), catalog)
        counter_dic[location] = tiles_counter

    return counter_dic


if __name__ == "__main__":

    old_flooduno_root = r"Y:\Users\Ruiyi\cleaned big dataset"
    flood_events = get_folders(old_flooduno_root)

    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    
    # There are 15 known flood events in the old version of unosat flood dataset
    assert len(flood_events) == 15

    event_dic = {}

    for event in tqdm(flood_events, leave=True):
        counter_dict = handle_event(old_flooduno_root, event, catalog)
        event_dic[event] = counter_dict

    # with open('statistics.json', 'w') as _out:
    #     json.dump(event_dic, _out)