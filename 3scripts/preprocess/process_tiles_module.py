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
from pyproj import Transformer
from geopy.geocoders import Nominatim
from pathlib import Path
import json
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray as rxr
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

def compress_geotiff_rasterio(input_tile_path, output_tile_path, compression="lzw"):
    with rasterio.open(input_tile_path) as src:
        profile = src.profile
        profile.update(compress=compression)

        with rasterio.open(output_tile_path, "w", **profile) as dst:
            dst.write(src.read())

# SELECT AND SPLIT TO TRAIN, VAL, TEST
def has_enough_valid_pixels(file_path, analysis_threshold, mask_threshold):
    """
    has mask layer?
    has analysis_extent layer?
    exceeds analysis_threshold?
    exceeds mask_threshold?
    returns True if both conditions are met
    defaults to False.
    TODO not capturing miss_ext if no mask layer
    """
    # print('+++has_enough_valid_pixels')
    # print('---analysis_threshold= ', analysis_threshold)
    # print('---mask_threshold= ', mask_threshold)

    try:
        with rasterio.open(file_path) as dataset:
            # Assuming the mask layer is identified by its name or index
            mask_layer = None
            analysis_extent_layer = None
            missing_extent = 0
            missing_mask = 0
            result = False

            for idx, desc in enumerate(dataset.descriptions):
                if desc == "mask":  # Find the 'mask' layer by its description
                    mask_layer = idx + 1  # Bands in rasterio are 1-based
                elif desc == "analysis_extent":
                    analysis_extent_layer = idx + 1

            total_pixels = dataset.width * dataset.height
            # READ MASK LAYER

            if not analysis_extent_layer:
                missing_extent = 1
            else:
                analysis_extent_data = dataset.read(analysis_extent_layer)
                if ((analysis_extent_data == 1).sum()) / total_pixels > analysis_threshold:
                    # print(f"---EXTENT pixels deficit: {file_path}")
                    result = True

            if not mask_layer:
                print(f"---No 'mask' layer found in {file_path}")
                missing_mask = 1
                result = False
            else:
                mask_data = dataset.read(mask_layer)
                if ((mask_data == 1).sum()) / total_pixels > mask_threshold:
                    # print(f"---MASK pixels deficit: {file_path}")
                    result = True

            return result, missing_extent, missing_mask
    except Exception as e:
        print(f"---Unexpected error: {e}")
        return 0,0,0  # Handle any other exceptions


        

def select_tiles_and_split(source_dir, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, MAKEFOLDER):
    print('+++select_tiles_and_split')
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

    with open(dest_dir / "train.txt", "w") as traintxt,  open(dest_dir / "val.txt", "w") as valtxt,  open(dest_dir / "test.txt", "w") as testtxt:
        # Get a list of all files in the source directory
        files = list(source_dir.glob('*'))  # Modify '*' if you want a specific file extension
        total_files = len(files)
        print(f"---Total files: {total_files}")

        # FILTER FILES BY VALID PIXELS
        # filtered_tiles = [file for file in tqdm(files) if has_enough_valid_pixels(file)]
        filtered_tiles = []
        for file in tqdm(files):
            # print(f"---Checking {file.name}")
            selected, missing_extent, missing_mask = has_enough_valid_pixels(file, analysis_threshold, mask_threshold)
            tot_missing_extent  += missing_extent
            tot_missing_mask += missing_mask
            if not selected:
                # print(f"---Rejected {file.name}")
                rejects += 1
            else:
                filtered_tiles.append(file)


        print(f"---Rejected files: {rejects}")
        # print(f"---Filtered files: {filtered_tiles}")

        if MAKEFOLDER:
            # SHUFFLE FILES
            random.shuffle(filtered_tiles)  # Shuffle files for random split

            # Calculate split indices
            train_end = int(len(files) * train_ratio)
            val_end = train_end + int(len(files) * val_ratio)

            # Split files into train, val, and test sets
            train_files = filtered_tiles[:train_end]
            val_files = filtered_tiles[train_end:val_end]
            test_files = filtered_tiles[val_end:]

            # Copy files to their respective folders
            for file in tqdm(train_files, desc="Copying train files"):
                try:
                    shutil.copy(file, train_dir / file.name)
                    traintxt.write(f"{file.name}\n")
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

            # print(f"---Total files: {len(files)}")
            # print(f"---Train files: {len(train_files)}")
            # print(f'---Test files: {len(test_files)}')
            # print(f"---Validation files: {len(val_files)}")
            assert abs(len(train_files) + len(val_files) + len(test_files) == len(files)) < 1e-6, "Files not split correctly"
            # with open("train.txt", "r") as tra, open("val.txt", "r") as val, open("test.txt", "r") as tes:
            with     open(dest_dir / "train.txt", "r") as traintxt,  open(dest_dir / "val.txt", "r") as valtxt,  open(dest_dir / "test.txt", "r") as testtxt:
            
                    # FINAL CHECK
                    if len(traintxt.readlines()) != len(train_files):
                        print('---train.txt not created successfully')
                        print(f'---{traintxt.readlines()}')
                    if len(valtxt.readlines()) != len(val_files):
                        print('---val.txt not created successfully')
                        print('---val.txt', valtxt.readlines())
                    if len(testtxt.readlines()) != len(test_files):
                        print('---test.txt not created successfully')
                        print('---test.txt', testtxt.readlines())
    return total_files, rejects, tot_missing_extent, tot_missing_mask

# NOT USED?
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



def tile_datacube(datacube_path, event, tile_size=256, stride=256):
    """
    Tile a DATASET (from which the dataarray is selected) and save to 'tiles' dir in same location.
    'ARGs:
      event is a Path object: datacube is a xarray object
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
        # Ensure 'valid' layer exists
        if 'valid' not in tile.coords['layer']:
            print("---Warning: 'valid' layer not found in tile.")
            return True  # If 'valid' layer is missing, treat as no valid data

        if 1 not in tile.sel(layer='valid').values:
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
    
    #print('---making tiles dir  in event dir')
    #print('---event= ', event.name)

    tile_path = Path(event, 'tiles') 
    os.makedirs(tile_path, exist_ok=True)
    # check new tiles dir exists
    datacube = rxr.open_rasterio(datacube_path)
    print('==============================')
    print('---opened datacube= ', datacube)
    print('==============================')

    print('---opened datacube crs = ', datacube.rio.crs)

    # Select the 'valid' layer within 'data1'
    valid_values = datacube['data1'].sel(layer='valid').values

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
    for y_idx in tqdm(range(num_y_tiles),desc="### Processing tiles by row"):

        for x_idx in range(num_x_tiles):

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, datacube.x.size)
            y_end = min(y_start + tile_size, datacube.y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            try:
                tile = datacube['data1'].sel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            except KeyError:
                print("---Error: 'data1' not found in datacube.")
                return

            # Skip empty tiles
            if tile.sizes["x"] == 0 or tile.sizes["y"] == 0:
                print("---Empty tile encountered; skipping.")
                continue
            # print('==============================')
            # print('---TILE= ', tile)
            # print('==============================')
            # print('--- 1 tile as dataarray before saving= ', tile)

            # CREAE A JSON FILE FOR EACH TILE
            create_tile_json(tile, event, x_idx, y_idx, tile_path)   

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

                continue

            if has_no_mask(tile):
                num_nomask +=1
                print('---tile has no mask')
                continue

            # TODO mask pixel check here + add to json

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


