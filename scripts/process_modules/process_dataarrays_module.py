import numpy as np
import rasterio
from pathlib import Path
import shutil
from tqdm import tqdm
import shutil
import random
import os
import xarray as xr
import json
import rasterio
import sys
import json

from pyproj import Transformer
from geopy.geocoders import Nominatim
from pathlib import Path
import json
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray as rxr
from rioxarray import open_rasterio
from rasterio.windows import Window
from scripts.process_modules.process_helpers import dataset_type, print_dataarray_info, nan_check, pad_tile


# NORMALIZE
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

def normalise_a_tile(file_path, output_path):
    # print('++++IN NORMALIZE A TILE')
    output_path = output_path / file_path.name
    if file_path.suffix.lower() not in ['.tif', '.tiff']:
        return
    # print(f"---Normalizing {file_path.name}")
    with rasterio.open(file_path) as src:
        meta = src.meta.copy()
        meta.update(dtype='float32')  # Ensure dtype can store normalized values

        with rasterio.open(output_path, 'w', **meta) as dst:
            for band in range(1, src.count + 1):
                # print(f"---normalizing band {band}")
                try:
                    band_name = src.descriptions[band - 1].lower()
                    data = src.read(band)

                    if band_name in ['vv','vh','grd','dem','slope' ] and np.min(data) != np.max(data) :
                        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                        dst.write(normalized_data.astype('float32'), band)
                        # print(f"---band {band_name} to be normalized")
                    else:
                        dst.write(data, band)
                        # print(f"---band {band_name} not normalized")
                    dst.set_band_description(band, band_name)


                except Exception as e:
                    print(f"---Error normalizing band {band}: {e}")
                    dst.write(data, band)

    # print(f"Normalized tile saved to {output_path}")


def normalize_inmemory_tile(tile):
    """
    Min-max normalize all layers in a tile except specified layers.
    """
    # print('+++in normalize_inmemory_tile')
    skip_layers =  ['mask', 'valid']
    normalized_tile = tile.copy()

    for layer in tile.coords['layer'].values:
        # print(f"---Normalizing layer '{layer}'")
        if layer in skip_layers:
            # print(f"---Skipping normalization for layer '{layer}'")
            continue
        
        layer_data = tile.sel(layer=layer)
        layer_min, layer_max = layer_data.min().item(), layer_data.max().item()
        # print(f"---Layer '{layer}': Min={layer_min}, Max={layer_max}")
        if layer_min != layer_max:  # Avoid division by zero
            normalized_tile.loc[dict(layer=layer)] = (layer_data - layer_min) / (layer_max - layer_min)
            normalized=True
        else:
            print(f"---Layers '{layer}' has constant values; skipping normalization")
            normalized=False
        return tile.astype('float32'), normalized

def log_clip_minmaxnorm_layer(tile):
    """
    Preprocess SAR data for U-Net segmentation.
    1. Log Transform
    2. Clip Extreme Values
    3. Normalize to [0, 1]
    """
    skip_layers = ['mask', 'valid']
    preprocessed_tile = tile.copy()

    for layer in tile.coords['layer'].values:
        if layer not in skip_layers:
            # Log Transform
            layer_data = np.log1p(tile.sel(layer=layer))
            # Clip Extreme Values
            lower_bound = np.percentile(layer_data, 2)
            upper_bound = np.percentile(layer_data, 98)
            layer_data = np.clip(layer_data, lower_bound, upper_bound)
            # Min-Max Normalize
            layer_min = layer_data.min().item()
            layer_max = layer_data.max().item()
            if layer_min != layer_max:
                preprocessed_tile.loc[dict(layer=layer)] = (layer_data - layer_min) / (layer_max - layer_min)
                normalized= True
            else:
                print(f"---Layer '{layer}' in tile '{tile.attrs['filename']}' is uniform; skipping normalization")
                normalized= False
    return preprocessed_tile.astype('float32'), normalized

def log_clip_minmaxnorm(tile, global_min, global_max):
    """
    Preprocess SAR data for U-Net segmentation using global min and max.
    1. Log Transform
    2. Clip Extreme Values
    3. Normalize to [0, 1]
    """
    # print('+++in log_clip_minmaxnorm')
    skip_layers = ['mask', 'valid']
    preprocessed_tile = tile.copy()

    for layer in tile.coords['layer'].values:
        if layer not in skip_layers:
            # Log Transform
            layer_data = np.log1p(tile.sel(layer=layer))

            # Clip Extreme Values
            lower_bound = np.percentile(layer_data, 2)  # This can still be computed if necessary
            upper_bound = np.percentile(layer_data, 98)
            layer_data = np.clip(layer_data, lower_bound, upper_bound)

            # Min-Max Normalize using global_min and global_max
            if global_min != global_max:
                preprocessed_tile.loc[dict(layer=layer)] = (layer_data - global_min) / (global_max - global_min)
                normalized = True
            else:
                print(f"---Global min and max are identical for layer '{layer}'; skipping normalization.")
                normalized = False
    return preprocessed_tile.astype('float32'), normalized



def calculate_global_min_max_nc(datacube, layer_name, percentile_min=2, percentile_max=98):
    """
    Calculate global min and max percentiles for dataset normalization from xarray.DataArray.
    Args:
        datacube (str): Path to the NetCDF file or DataArray.
        layer_name (str): Name of the layer to use for normalization (e.g., "HH").
        percentile_min (int): Lower percentile for clipping.
        percentile_max (int): Upper percentile for clipping.
    Returns:
        tuple: (global_min, global_max)
    """
    print('+++in calculate_global_min_max_nc')
    print('---layer_name= ', layer_name)
    
    # Open dataset
    ds = xr.open_dataset(datacube)
    print('---ds= ', ds)
    var = list(ds.data_vars)[0]
    da = ds[var]

    print('---da layers= ', da.coords["layer"].values)
    
    # Check if the required layer exists
    if layer_name not in da.coords["layer"].values:
        raise ValueError(f"Layer '{layer_name}' not found in dataset. Available layers: {list(da.coords['layer'].values)}")
    
    # Extract layer data
    layer_data = da.sel(layer=layer_name)  # Extract the variable as a NumPy array
    print(f"Extracted data for layer '{layer_name}', shape: {layer_data.shape}")
    
    # Flatten pixel values
    all_pixel_values = layer_data.values.flatten()  # Access NumPy array and flatten
    print(f"Flattened pixel values, total: {len(all_pixel_values)}")
    
    # Compute percentiles
    global_min = np.percentile(all_pixel_values, percentile_min)
    global_max = np.percentile(all_pixel_values, percentile_max)
    print(f"Calculated global min={global_min}, max={global_max}")
    
    return global_min, global_max



def get_global_min_max(data_path, layer_name, min_max_file=None, percentile_min=2, percentile_max=98):
    """
    Calculate or load global min and max for dataset normalization.
    Args:
        data_path (str): Path to the dataset directory.
        layer_name (str): Layer to use for normalization.
        min_max_file (str): Path to the file storing min-max values.
        percentile_min (int): Lower percentile for clipping.
        percentile_max (int): Upper percentile for clipping.
    Returns:
        tuple: (global_min, global_max)
    """
    print('+++in get_global_min_max')
    print('---data_path= ', data_path)
    print('---layer_name= ', layer_name)
    # Check if the file exists
    if min_max_file.exists():
        # Load existing min-max values
        with open(min_max_file, "r") as f:
            global_min, global_max = map(float, f.readline().split(","))
        print(f"Loaded global min-max from {min_max_file}: Min={global_min}, Max={global_max}")
    else:
        # Calculate min-max values
        global_min, global_max = calculate_global_min_max_nc(data_path, layer_name, percentile_min, percentile_max)
        # Save to file
        with open(min_max_file, "w") as f:
            f.write(f"{global_min},{global_max}")
        print(f"Calculated and saved global min-max: Min={global_min}, Max={global_max}")
    
    stats = (global_min, global_max)
    return stats



# SELECT AND SPLIT
def has_enough_valid_pixels(file_path, analysis_threshold, mask_threshold):
    """
    has mask layer?
    exceeds analysis_threshold?
    exceeds mask_threshold?
    returns REJECTED = 1 if any of the above is NOT true
    has analysis_extent layer?
    """
    # print('\n+++has_enough_valid_pixels')
    # print('---file_path= ', file_path.name)
    # print('---analysis_threshold= ', analysis_threshold)
    # print('---mask_threshold= ', mask_threshold)
    if file_path.suffix.lower() not in ['.tif', '.tiff']:
        print(f"---Skipping {file_path.name}: not a TIFF file")
        return 0,0,0
    else:
        try:
            with rasterio.open(file_path) as dataset:
                # Assuming the mask layer is identified by its name or index
                mask_layer = None
                analysis_extent_layer = None
                missing_extent = 0
                missing_mask = 0
                rejected = 1
                # print('---rejected= ', rejected)    

                for idx, desc in enumerate(dataset.descriptions):
                    if desc == "mask":  # Find the 'mask' layer by its description
                        mask_layer = idx + 1  # Bands in rasterio are 1-based
                    elif desc == "analysis_extent":
                        analysis_extent_layer = idx + 1

                total_pixels = dataset.width * dataset.height
                # ANALYSIS EXTENT
                if analysis_extent_layer:
                    analysis_extent_data = dataset.read(analysis_extent_layer)
                    # print('---analysis_extent_data= ', (analysis_extent_data == 1).sum())
                    if ((analysis_extent_data == 1).sum()) / total_pixels >= analysis_threshold:
                        # print(f"---EXTENT pixels deficit: {file_path}")
                        rejected = 0  
                    else:
                        rejected = 1
                else:
                    missing_extent = 1
                # print('---rejected= ', rejected)

                # MASK
                if  mask_layer:
                    mask_data = dataset.read(mask_layer)
                    if ((mask_data == 1).sum()) / total_pixels >= mask_threshold:
                        # print(f"---MASK pixels deficit: {file_path}")
                        rejected = 0
                    else:
                        rejected = 1
                else:
                    missing_mask = 1
                # print('---final rejected= ', rejected)
                return rejected, missing_extent, missing_mask
        except Exception as e:
            print(f"---Unexpected error: {e}")
            return 0,0,0  # Handle any other exceptions

def select_tiles_and_split(source_dir, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, MAKEFOLDER):
    '''
    TODO return  where selected files is below some threshold or zero
    '''
    print('\n+++select_tiles_and_split')
    # Ensure the ratios sum to 1.0
    # assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    # Create destination folders
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"

    rejects = 0
    tot_missing_extent = 0
    tot_missing_mask = 0

    with open(dest_dir / "train.txt", "a") as traintxt,  open(dest_dir / "val.txt", "a") as valtxt,  open(dest_dir / "test.txt", "a") as testtxt:
        # Get a list of all files in the source directory
        files = list(source_dir.glob('*'))  # Modify '*' if you want a specific file extension
        total_files = len(files)
        # print(f"---Total files: {total_files}")

        # FILTER FILES BY VALID PIXELS
        selected_tiles = []
        for file in tqdm(files, desc=f"Filtering files for {source_dir.name}", unit="file"):
            # print(f"---Checking {file.name}")
            rejected, missing_extent, missing_mask = has_enough_valid_pixels(file, analysis_threshold, mask_threshold)
            tot_missing_extent  += missing_extent
            tot_missing_mask += missing_mask
            if rejected:
                rejects += 1
            else:
                selected_tiles.append(file)

        # print(f"---Filtered files: {selected_tiles}")

        if MAKEFOLDER:
            # SHUFFLE FILES
            random.shuffle(selected_tiles)  # Shuffle files for random split

            # Calculate split indices
            train_end = int(len(selected_tiles) * train_ratio)
            val_end = train_end + int(len(selected_tiles) * val_ratio)

            # Split files into train, val, and test sets
            train_files = selected_tiles[:train_end]
            val_files = selected_tiles[train_end:val_end]
            test_files = selected_tiles[val_end:]

            # Copy files to their respective folders
            for file in tqdm(train_files, desc="Copying train files"):
                try:
                    shutil.copy(file, train_dir / file.name)
                    # print(f"---{file.name} copied to {train_dir}")
                    traintxt.write(f"{file.name}\n")
                    # print(f"---{file.name} written to train.txt")
                except Exception as e:
                    print(f"Error copying {file}: {e}")

            for file in tqdm(val_files, desc="Copying validation files"):
                try:
                    shutil.copy(file, val_dir / file.name)
                    valtxt.write(f"{file.name}\n")
                except Exception as e:
                    print(f"---Error copying {file}: {e}")

            for file in tqdm(test_files, desc="Copying test files"):
                try:
                    shutil.copy(file, test_dir / file.name)
                    testtxt.write(f"{file.name}\n")
                except Exception as e:
                    print(f"---Error copying {file}: {e}")

                        # Flush the output to ensure data is written to disk
            traintxt.flush()
            valtxt.flush()
            testtxt.flush()
            # print(f'---folder total files: {total_files}')
            if len(selected_tiles) < 10:
                print(f"#######\n---{source_dir.parent.name} selected tiles: {len(selected_tiles)}\n#######")
            # print(f"---folder train files: {len(train_files)}")
            # print(f'---folder test files: {len(test_files)}')
            # print(f"---folder validation files: {len(val_files)}")
            assert int(len(train_files)) + int(len(val_files)) + int(len(test_files)) == int(len(selected_tiles)), "Files not split correctly"

            print('---END OF SPLI FUNCTION--------------------------------')
    return total_files, selected_tiles, rejects, tot_missing_extent, tot_missing_mask


def copy_data_and_generate_txt(data_folders, destination):
    """Copy all files into a centralized destination and create .txt files for train, val, and test."""
    print('+++in copy_data_and_generate_txt')
    dest_paths = {key: destination / key for key in data_folders}
    txt_files = {key: destination / f"{key}.txt" for key in data_folders}
    
    # Ensure destination folders exist
    for path in dest_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Process each folder type (train, val, test)
    for key, folders in data_folders.items():
        with open(txt_files[key], 'w') as txt_file:
            for folder in folders:
                file_list = list(folder.glob('*'))  # Get all files in the folder
                for file_path in tqdm(file_list, desc=f"Copying {key} files from {folder}", unit="file"):
                    if file_path.is_file():  # Copy only files
                        dest_file_path = dest_paths[key] / file_path.name
                        shutil.copy(file_path, dest_file_path)
                        # Write the path of the copied file to the .txt file
                        txt_file.write(str(dest_file_path) + '\n')
        print(f"Copied {key} files and generated {key}.txt")

# CREATE STAC METADATA
def create_stac_metadata(tile, tile_id, output_dir, parent_dataset_id="datacube_01"):
    """
    Create STAC JSON metadata for a tile.
    :param tile: xarray object representing the tile.
    :param tile_id: Unique ID for the tile.
    :param output_dir: Directory to save the JSON file.
    :param parent_dataset_id: ID of the parent dataset.
    """
    # Extract bounding box (assuming tile.coords contains x and y dims)
    bbox = [
        tile.coords['x'].min().item(),  # Min x
        tile.coords['y'].min().item(),  # Min y
        tile.coords['x'].max().item(),  # Max x
        tile.coords['y'].max().item()   # Max y
    ]
    
    # General tile statistics
    overall_stats = {
        "min": float(tile.data.min()),
        "max": float(tile.data.max()),
        "mean": float(tile.data.mean()),
        "std": float(tile.data.std()),
        "unique_values": np.unique(tile.data).tolist()
    }

    # Layer-specific statistics
    layer_stats = {}
    for layer in tile.coords['layer'].values:
        layer_data = tile.sel(layer=layer).values
        layer_stats[layer] = {
            "min": float(layer_data.min()),
            "max": float(layer_data.max()),
            "mean": float(layer_data.mean()),
            "std": float(layer_data.std()),
            "percentage_1s": float(np.count_nonzero(layer_data == 1) / layer_data.size * 100)
        }
    
    # Construct STAC metadata
    metadata = {
        "id": tile_id,
        "type": "Feature",
        "bbox": bbox,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[0], bbox[1]]
            ]]
        },
        "properties": {
            "parent_dataset_id": parent_dataset_id,
            "statistics": {
                "overall": overall_stats,
                "layers": layer_stats
            }
        },
        "assets": {
            "tile_data": {
                "href": f"{output_dir}/{tile_id}.tif",
                "type": "image/tiff"
            }
        }
    }

    # Save JSON to file
    json_path = f"{output_dir}/{tile_id}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"STAC metadata saved to {json_path}")

# REPROJECT
def location_to_crs(location, crs):

    
    # Define target CRS as EPSG:4326
    dst_crs = CRS.from_epsg(4326)  # WGS84 EPSG:4326
    
    # Calculate transform and new dimensions for the reprojected dataset
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    
    # Update metadata for the new file
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    # Define output path if not provided
    if output_path is None:
        input_path = Path(src.name)  # Original file path
        output_path = input_path.with_name(f"epsg4326_{input_path.name}")

    # Reproject and save to the new file
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

    return output_path  # Return the path of the reprojected file


# TILING
def tile_datacube_rxr(datacube_path, save_tiles_path, tile_size, stride, norm_func, stats, inference=False):
    """
    Tile a DATASET (extracted from a dataarray is selected) and save to 'tiles' dir in same location.
    'ARGs:
    
    play with the stride and tile_size to get the right size of tiles.
    TODO add date etc to saved tile name?
    TODO add json file with metadata for each tile inv mask and anal_ext pixel count etc
    """
    print('+++++++in tile_datacube rxr fn ++++++++')
    global_min, global_max = stats
    num_tiles = 0
    num_saved = 0
    num_has_nans = 0
    num_novalid_layer = 0
    num_novalid_pixels = 0
    num_nomask = 0
    num_nomask_pixels = 0
    num_failed_norm = 0
    num_not_256 = 0


    ds = xr.open_dataset(datacube_path)
    var = list(ds.data_vars)[0]
    da = ds[var]

    print('---DS VARS BEFORE TILING = ', list(ds.data_vars))

    # print_dataarray_info(da)
    if da.chunks:
        for dim, chunk in zip(da.dims, da.chunks):
            print(f"---Dimension '{dim}' has chunk sizes: {chunk}")
    tile_metadata = []
    inference_tiles = []

    print('----START TILING----------------')
    for y_start in tqdm(range(0, da.y.size, stride), desc="### Processing tiles by row"):
        for x_start in range(0, da.x.size, stride):

            # Ensure tiles start on the boundary and fit within the dataset
            x_end = min(x_start + tile_size, da.x.size)
            y_end = min(y_start + tile_size, da.y.size)


            # Select the subset of data for the current tile
            try:
                tile = da.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            except KeyError:
                print("---Error: tiling.")
                return
            
            num_tiles += 1    

            # print('---TILE INFO AT START')
            # print_dataarray_info(tile)
            if not inference:
                if has_no_mask_pixels(tile):
                    num_nomask_pixels +=1
                    # print('---tile has no mask pixels')
                    continue

            if int(tile.sizes["x"]) != tile_size or int(tile.sizes["y"]) != tile_size:
                # print("---odd shaped encountered; padding.")
                # padtile = pad_tile(tile, 250)
                # print(f"---Tile dimensions b4 padding: {tile.sizes['x']}x{tile.sizes['y']}")
                tile = pad_tile(tile, tile_size)
                # print("---Tile coords:", tile.coords)
                # print(f"---Tile dimensions after padding: {tile.sizes['x']}x{tile.sizes['y']}")
                num_not_256 += 1

            # #   FILTER OUT TILES WITH NO DATA
            # # print('---filtering out bad tiles')
            if contains_nans(tile):
                num_has_nans += 1
                print('---tile has nans')
                continue

            # if has_no_valid_layer(tile):
            #     num_novalid_layer += 1
            #     # print('---tile has no valid layer')
            #     continue
            # if has_no_valid_pixels(tile):
            #     num_novalid_pixels += 1
            #     # print('---tile has no valid pixels')
            #     continue
            # if has_no_mask(tile):
            #     num_nomask +=1
            #     # print('---tile has no mask')
            #     continue

            # print("---PRINTING DA INFO B4 NORM-----")
            # print_dataarray_info(tile)
            normalized = False
            # normalized_tile, normalized = normalize_inmemory_tile(tile)
            # print('---norm_func= ', norm_func)  
            if norm_func == 'logclipmm_g':
                normalized_tile, normalized = log_clip_minmaxnorm(tile, global_min, global_max) 
        
            if not normalized:
                print('---Failed to normalize tile')
                num_failed_norm += 1
                continue

            tile_name = f"tile_{datacube_path.parent.name}_{x_start}_{y_start}.tif"

            dest_path = save_tiles_path  / tile_name
            print
            if not dest_path.parent.exists():
                os.makedirs(dest_path.parent, exist_ok=True)
                print('---created dir= ', dest_path.parent)

            ######### SAVE TILE ############
            # Save layer names as metadata
            layer_names = list(normalized_tile.coords["layer"].values)
            layer_names = [str(name) for name in layer_names]
            # get crs and transform
            crs = normalized_tile.rio.crs
            transform = normalized_tile.rio.transform()
            tile_data = normalized_tile.values
            num_layers, height, width = tile_data.shape
            # print('---layer_names= ', layer_names)
            with rasterio.open(dest_path, 'w',
                                driver='GTiff',
                                height=height,
                                width=width,
                                count=num_layers,
                                dtype=tile_data.dtype,
                                crs=crs,
                                transform=transform,
                                compress=None) as dst:
                for i in range(1, num_layers + 1):
                    print('---num_layers= ', num_layers)
                    dst.write(tile_data[i - 1], i)
                    dst.set_band_description(i, layer_names[i-1])  # Add band descriptions
            num_saved += 1
            inference_tiles.append(normalized_tile)
            # Store metadata for stitching
            tile_metadata.append({
                "tile_name": tile_name,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end
            })
            # Save metadata for stitching
            metadata_path = save_tiles_path / "tile_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(tile_metadata, f, indent=4)
            # print(f"---Saved metadata to {metadata_path}")

    if inference:
        return inference_tiles, tile_metadata            

        #     if num_saved  == 100:
        #         break
        # if num_saved  == 100:
        #     break
    print('--- num_tiles= ', num_tiles)
    print('---num_saved= ', num_saved)  
    print('---num_has_nans= ', num_has_nans)
    print('---num_novalid_layer= ', num_novalid_layer)
    print('---num_novalid_pixels= ', num_novalid_pixels)
    print('---num_nomask= ', num_nomask)
    print('---num_nomask_pixels= ', num_nomask_pixels)
    print('---num_failed_norm= ', num_failed_norm)
    return num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask, num_nomask_pixels, num_failed_norm, num_not_256

def tile_datacube_rasterio(datacube_path, save_tiles_path, tile_size=256, stride=256):
    """
    Tile a DATASET (extracted from a dataarray is selected) and save to 'tiles' dir in same location.
    'ARGs:
    
    play with the stride and tile_size to get the right size of tiles.
    TODO add date etc to saved tile name?
    TODO add json file with metadata for each tile inv mask and anal_ext pixel count etc
    """
    print('\n+++++++in tile_datacube fn ++++++++')

    # datacube = rxr.open_rasterio(datacube_path, variable='data1') # OPENS A DATAARRAY

    # EXTRACT THE DATAARRAY WE WANT
    datacube = xr.open_dataset(datacube_path) # OPENS A DATASET
    datacube = datacube['data1']

    print('==============================')
    print('---opened datacube= ', datacube)
    print('==============================')
    # check if datacube is a dataarray or dataset
    if isinstance(datacube, xr.Dataset):
        print('---datacube is a dataset')
    else:
        print('---datacube is a dataarray')
        # datacube = datacube.to_array(dim='layer', name='data1')

    print("---Raw Datacube Check----------------")
    layer_names = []
    for l in datacube.coords["layer"].values:
        layer_names.append(str(l))

    print('---layers= ', layer_names)

    # print('---opened datacube crs1 = ', datacube.rio.crs)
    datacube.rio.write_crs("EPSG:4326", inplace=True)
    # print('---opened datacube crs2 = ', datacube.rio.crs)



    num_tiles = 0
    num_saved = 0
    num_has_nans = 0
    num_novalid_layer = 0
    num_novalid_pixels = 0
    num_nomask = 0
    num_nomask_pixels = 0



    with rasterio.open(datacube_path) as src:
        # print src dtype
        print('---src dtype:', src.dtype)


        for y_start in tqdm(range(0, src.height, stride), desc="### Processing tiles by row"):
            for x_start in range(0, src.width, stride):
                x_start = x_start * stride
                y_start = y_start * stride
                window = Window(x_start, y_start, tile_size, tile_size)

                # Ensure the tile stays within the bounds of the dataset
                x_end = min(x_start + tile_size, src.width)
                y_end = min(y_start + tile_size, src.height)
                width = x_end - x_start
                height = y_end - y_start
                window = Window(x_start, y_start, width, height)
                print('---window= ', window)

                # Read the data for this tile
                tile_data = src.read(window=window)
                num_tiles += 1
                # print('---*****num_tiles*************= ', num_tiles)

                if  num_tiles % 250 == 0:
                    print(f"Counted { num_tiles} tiles")

                #   FILTER OUT TILES WITH NO DATA
                # print('---filtering out bad tiles')

                




                if contains_nans(src):
                    num_has_nans += 1
                    # print('---src has nans')
                    continue
                if has_no_valid_layer(src):
                    num_novalid_layer += 1
                    # print('---src has no valid layer')
                    continue
                if has_no_valid_pixels(src):
                    num_novalid_pixels += 1
                    # print('---src has no valid pixels')
                    continue
                if has_no_mask(src):
                    num_nomask +=1
                    # print('---src has no mask')
                    continue
                if has_no_mask_pixels(src):
                    num_nomask_pixels +=1
                    # print('---src has no mask pixels')
                    continue
                # Save the tile as GeoTIFF
                tile_path = save_tiles_path / f"tile_{y_start}_{x_start}.tif"
                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=tile_data.dtype,
                    crs=src.crs,
                    transform=rasterio.windows.transform(window, src.transform),
                ) as dst:
                    # FILTER OUT TILES WITH NO DATA
                    print('---filtering out bad tiles')


                    for i in range(1, src.count + 1):
                        dst.write(tile_data[i - 1], i)  # Write each band
                        dst.set_band_description(i, layer_names[i-1])  # Add band descriptions
                    
                    # Normalize the tile
                    # print('---normalizing tile')
                    dst = normalize_inmemory_tile(dst)

                print("Tiling complete.")

                # print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
                # print('---TILE= ', tile)
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')


                num_saved += 1

            # OPEN SAVED TILE AND CHECK VALUES IN LAYERS
            print('-----------------------------')
            print('---SAVED TILE CHECK-------')

            # saved_tile = rxr.open_rasterio(dest_path)
            # print('---saved_tile= ', saved_tile)
            # for i in saved_tile.coords["band"].values:
            #     # Select the second band (e.g., band 2)
            #     # Find unique values
            #     vals = np.unique(saved_tile.sel(band=i))
            #     print(f"{i}: num Unique values: {len(vals)}")

            with rasterio.open(tile_path) as src:
                print('---src:', src.count)
                for i in range(1, src.count + 1):
                    band_name = src.descriptions[i - 1].lower()
                    print('---bandname ',band_name)
                    return
            print('-----------------------------')

    # print('--- num_tiles= ', num_tiles)
    # print('---num_saved= ', num_saved)  
    # print('---num_has_nans= ', num_has_nans)
    # print('---num_novalid_layer= ', num_novalid_layer)
    # print('---num_novalid_pixels= ', num_novalid_pixels)
    # print('---num_nomask= ', num_nomask)
    # print('---num_nomask_pixels= ', num_nomask_pixels)
    return num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask, num_nomask_pixels

# TILE CHECKS

def contains_nans(tile):
    """
    Checks if specified bands contain NaN values.
    Returns True if NaNs are found, along with the count of NaNs per band.
    Returns False if no NaNs are found.
    """
    contains_nan = False  # Flag to track if any NaNs are found
    for band in tile.coords['layer'].values:
        # Calculate the number of NaNs in the current band
        nan_count = np.isnan(tile.sel(layer=band)).sum().values
        if nan_count > 0:
            print(f"Tile contains {nan_count} NaN values in {band}")
            contains_nan = True
    return contains_nan  # True if any NaNs were found, False if none

def has_no_valid_layer(tile):
    '''
    Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
    return False = ok
    '''
    # Ensure 'valid' layer exists
    # print("---Warning: 'valid' layer not found in tile.")
    return 'valid' not in tile.coords['layer'].values  # If valid data is found, return False
    
def has_no_valid_pixels(tile, ):

    #print("Values in 'valid' layer for tile:", tile.sel(layer='valid').values)
    return 1 not in tile.sel(layer='valid').values  # If valid data is found, return False

def has_no_mask(tile):
    '''
    Filters out tiles without mask.
    return False = ok
    '''
    return 'mask' not in tile.coords['layer'].values

def has_no_mask_pixels(da):

    mask_values = da.sel(layer='mask').values
    max = mask_values.max().item()
    if max != 1:
        # print(f"---Mask layer contains no valid pixels (max value: {max})")
        return True
    return False

def is_not_256(data):
    if data.shape[0] != 256 or data.shape[1] != 256:
        # print('---padding')
        # pad_width = 256 - data.shape[1]
        # pad_height = 256 - data.shape[0]
        # data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='reflect')
        return False
    else:
        return True



def check_novalues(path_to_tiff):
    with rasterio.open(path_to_tiff) as src:
        # Read the data from the first band
        band1 = src.read(1)

        # Check unique pixel values
        unique_values, counts = np.unique(band1, return_counts=True)
        print("Unique pixel values:", unique_values)
        print("Counts of each value:", counts)

def remove_nodata_from_tiff(input_tiff, output_tiff):
    """Remove the NoData flag from a TIFF, ensuring all pixel values are valid."""
    with rasterio.open(input_tiff) as src:
        # Copy the metadata and remove the 'nodata' entry
        profile = src.profile.copy()
        
        # Remove the NoData value from the profile
        if profile.get('nodata') is not None:
            print(f"Original NoData value: {profile['nodata']}")
            profile['nodata'] = None
        else:
            print("No NoData value set.")
        
        # Create a new TIFF without NoData
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            for i in range(1, src.count + 1):  # Loop through each band
                data = src.read(i)
                dst.write(data, i)
        
    print(f"Saved new TIFF without NoData to {output_tiff}")

# UTILITIES

def get_incremental_filename(base_dir, base_name):
    """
    Generates a unique directory name by adding an incremental suffix.
    Example: 'base_name', 'base_name_1', 'base_name_2', etc.
    
    :param base_dir: The parent directory.
    :param base_name: The base name of the directory.
    :return: A Path object for the unique directory.
    """
    dest_dir = base_dir / base_name
    counter = 1
    while dest_dir.exists():
        dest_dir = base_dir / f"{base_name}_{counter}"
        counter += 1
    return dest_dir


def make_train_folders(dest_dir):
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir, test_dir

def handle_interrupt(signal, frame):
    '''
    usage: signal.signal(signal.SIGINT, handle_interrupt)
    '''
    print("Interrupt received! Cleaning up...")
    # Add any necessary cleanup code here (e.g., saving model checkpoints)
    sys.exit(0)

def pad_tile(tile, target_size):
    # Get the current dimensions
    current_x = int(tile.sizes["x"])
    current_y = int(tile.sizes["y"])
    
    pad_x = target_size - current_x
    pad_y = target_size - current_y

    # Ensure padding is non-negative
    if pad_x < 0 or pad_y < 0:
        raise ValueError("Target size must be larger than current dimensions")

    # Calculate padding for both dimensions
    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    # Pad the data using numpy
    padded_data = np.pad(
        tile.data,  # Extract the underlying numpy array
        pad_width=((0, 0),  # No padding for the layer dimension
                   (pad_y_before, pad_y_after),  # Padding for y dimension
                   (pad_x_before, pad_x_after)),  # Padding for x dimension
        mode='reflect'
    )

    # Calculate resolution from coordinates if `res` is not present
    if 'res' in tile.attrs:
        res = tile.attrs['res']
    else:
        # Calculate resolution from coordinate spacing
        res_x = (tile.coords["x"].values[-1] - tile.coords["x"].values[0]) / (current_x - 1)
        res_y = (tile.coords["y"].values[-1] - tile.coords["y"].values[0]) / (current_y - 1)
        res = min(res_x, res_y)  # Use the smaller resolution as the default
    
    # Update coordinates
    new_x = np.linspace(
        float(tile.coords["x"].values[0]) - pad_x_before * res,
        float(tile.coords["x"].values[-1]) + pad_x_after * res,
        target_size
    )
    new_y = np.linspace(
        float(tile.coords["y"].values[0]) - pad_y_before * res,
        float(tile.coords["y"].values[-1]) + pad_y_after * res,
        target_size
    )
    
    # Create new DataArray with updated data and coordinates
    padded_tile = xr.DataArray(
        padded_data,
        dims=("layer", "y", "x"),
        coords={"layer": tile.coords["layer"], "y": new_y, "x": new_x},
        attrs=tile.attrs  # Preserve the original attributes
    )
    
    return padded_tile



# MAYBE NOT NEEDED
 
def compress_geotiff_rasterio(input_tile_path, output_tile_path, compression="lzw"):
    with rasterio.open(input_tile_path) as src:
        profile = src.profile
        profile.update(compress=compression)

        with rasterio.open(output_tile_path, "w", **profile) as dst:
            dst.write(src.read())

def find_data_folders(base_path):
    """Recursively search  to find 'train', 'val', and 'test' folders.
    create lists of paths to each folder type."""
    print('+++in find_data_folders')
    data_folders = {'train': [], 'val': [], 'test': []}
    for root, dirs, _ in os.walk(base_path):
        # print('---root = ', root)
        for d in dirs:
            if d in data_folders:
                data_folders[d].append(Path(root) / d)
    # print('---data_folders = ', data_folders)
    return data_folders
