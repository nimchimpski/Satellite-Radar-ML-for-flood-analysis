import os
import rasterio
import rioxarray # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from tqdm import tqdm
# import cv2
# import matplotlib.pyplot as plt
from pathlib import Path

VERSION = "UNOSAT-AI-v2"
TRACKING = "UNOSAT-AI-v1"


def custom_normalize(array, lower_percentile=2, upper_percentile=98, clip_range=(0, 1)):
    """
    Normalizes a given array by scaling values between the given percentiles
    and clipping to the specified range.

    Args:
    - array: Input data (xarray.DataArray or NumPy array)
    - lower_percentile: The lower bound for normalization (default 2)
    - upper_percentile: The upper bound for normalization (default 98)
    - clip_range: Tuple specifying the range to clip the normalized data (default (0, 1))

    Returns:
    - normalized_array: The normalized data array (same type as input)
    """
    print('+++in custom_normalize fn')
    print('---array type',type(array))
    
    # Check if the input is an xarray.DataArray
    if isinstance(array, xr.DataArray):
        print('---array is xarray.DataArray')

        # Rechunk 'y' dimension into a single chunk, leave other dimensions unchanged
        array = array.chunk({'y': -1})  

        # Apply normalization while preserving metadata
        min_val = array.quantile(lower_percentile / 100.0)
        max_val = array.quantile(upper_percentile / 100.0)

        # Normalize the array
        normalized_array = (array - min_val) / (max_val - min_val)

        # Clip values to the specified range
        normalized_array = normalized_array.clip(min=clip_range[0], max=clip_range[1])
        normalized_array = normalized_array.astype('float32')
        return normalized_array

    else:
        print('---array is not xarray.DataArray')
        # Handle as a NumPy array if not xarray.DataArray
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

# TODO check this...
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

def normalize_and_stack(data_root):
    '''
    Loop through each dataset to normalize and load into xarray datacube (stacked on a new dimension)
    'Event' is a single folder conatining flood event data     
    '''
    layers = []
    layer_names = []

    for event in data_root.iterdir():
        if event.is_dir():
            print(f"---Preprocessing event: {event.name}")
            # Get the datas info from the folder
            datas = make_datas(event)

        for tif_file, band_name in datas.items():
            try:
                print(f"---Loading {tif_file}")
                # Use Dask chunking for large .tif files
                stack = rioxarray.open_rasterio(Path(data_root, event, tif_file), chunks={'x' : 1024, 'y' : 1024})
            except Exception as e:
                print(f"---Error loading {tif_file}: {e}")
                continue  # Skip this file if there is an issue loading
            # Handle multi-band image data (e.g., VV, VH bands)
            if 'img.tif' in tif_file:
                try:
                    stack = stack.assign_coords({'band': band_name})  # Assign 'vv' and 'vh'
                    # Normalize both VV and VH bands directly on the xarray.DataArray

                    vv_norm = stack.sel(band=band_name[0])
                    vh_norm = stack.sel(band=band_name[1])
                    # Concatenate VV and VH bands along the 'band' dimension
                    norm_stack = xr.concat([vv_norm, vh_norm], dim='band')
                    layers.append(norm_stack)
                    layer_names.extend(band_name)  # ['vv', 'vh']   
                    print(f"---Successfully processed IMG layers: {band_name}")
                except Exception as e:
                    print(f"---Error normalizing IMG bands for {tif_file}: {e}")
                    continue    
            # Handle single-band layers (e.g., DEM, slope, mask)   
            elif 'elevation' or 'slope' in tif_file:
                try:
                    norm_stack = custom_normalize(stack)
                    layers.append(norm_stack)
                    layer_names.append(band_name)
                    print(f"---Successfully processed layer: {band_name}")
                except Exception as e:
                    print(f"---Error normalizing bands for {tif_file}: {e}")
                    continue
        
    if layers:
        # Convert data arrays to float32 before concatenation
        layers = [layer.astype('float32') for layer in layers]
        # Concatenate all layers along a new dimension called 'layer' or 'band'
        datacube = xr.concat(layers, dim='layer')

        # Assign the layer names (e.g., ['vv', 'vh', 'dem', 'slope']) to the 'layer' dimension
        datacube = datacube.assign_coords(layer=layer_names)
        return datacube
    else:
        print('---No layers found')
        return None 

def make_datas(event):
    datas = {}
    print(f"---gatting datas info from: {event.name}")
    for file in event.iterdir():
        # print(f'---file {file}')
        if 'elevation.tif' in file.name:
            # print(f'---elevation file found {file}')    
            datas[file.name] = 'dem'
        elif 'slope.tif' in file.name:
            # print(f'---slope file found {file}')    
            datas[file.name] = 'slope'
        elif 'msk.tif' in file.name:
            # print(f'---mask file found {file}')    
            datas[file.name] = 'mask'   
        elif 'valid.tif' in file.name:
            # print(f'---valid file found {file}')    
            datas[file.name] = 'analysis_extent'
        elif 'img.tif' in file.name:
            # print(f'---image file found {file}')    
            datas[file.name] = ['vv', 'vh']
    print('---datas ',datas)
    return datas
    
def main():
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    normalize_and_stack(data_root)
    # tile_statistics = {}

if __name__ == "__main__":
    main()

       