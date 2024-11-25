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
from pyproj import Transformer
from geopy.geocoders import Nominatim
from pathlib import Path
import json
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray as rxr
from rasterio.windows import Window


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
    skip_layers =  ['mask', 'valid']
    normalized_tile = tile.copy()

    for layer in tile.coords['layer'].values:
        if layer in skip_layers:
            print(f"---Skipping normalization for layer '{layer}'")
            continue
        
        layer_data = tile.sel(layer=layer)
        layer_min, layer_max = layer_data.min().item(), layer_data.max().item()
        if layer_min != layer_max:  # Avoid division by zero
            normalized_tile.loc[dict(layer=layer)] = (layer_data - layer_min) / (layer_max - layer_min)

            normalized_tile.loc[dict(layer=layer)] = (layer_data - layer_min) / (layer_max - layer_min)

    return normalized_tile


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
def tile_datacube_rxr(datacube_path, save_tiles_path, tile_size=256, stride=256):
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

    # print("---Raw Datacube Check----------------")
    # for layer in datacube.coords["layer"].values:
    #     layer_data = datacube.sel(layer=layer)
    #     print(f"Layer '{layer}': Min={layer_data.min().item()}, Max={layer_data.max().item()}")
    #     print(f"Layer '{layer}': Unique values: {np.unique(layer_data.values)}")

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

    for y_start in tqdm(range(0, datacube.y.size, stride), desc="### Processing tiles by row"):
        for x_start in range(0, datacube.x.size, stride):
            if num_saved == 1:
                break
            # Ensure tiles start on the boundary and fit within the dataset
            x_end = min(x_start + tile_size, datacube.x.size)
            y_end = min(y_start + tile_size, datacube.y.size)

            # Select the subset of data for the current tile
            try:
                tile = datacube.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            except KeyError:
                print("---Error: tiling.")
                return

            # Skip empty tiles
            if tile.sizes["x"] == 0 or tile.sizes["y"] == 0:
                print("---Empty tile encountered; skipping.")
                continue
            # print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
            # print('---TILE= ', tile)
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

            #CREAE A JSON FILE FOR EACH TILE
            # create_tile_json(tile, event, x_idx, y_idx, tile_path)   

            num_tiles += 1
            # print('---*****num_tiles*************= ', num_tiles)

            if  num_tiles % 250 == 0:
                print(f"Counted { num_tiles} tiles")

            #   FILTER OUT TILES WITH NO DATA
            # print('---filtering out bad tiles')
            if contains_nans(tile):
                num_has_nans += 1
                # print('---tile has nans')
                continue
            if has_no_valid_layer(tile):
                num_novalid_layer += 1
                # print('---tile has no valid layer')
                continue
            if has_no_valid_pixels(tile):
                num_novalid_pixels += 1
                # print('---tile has no valid pixels')
                continue
            if has_no_mask(tile):
                num_nomask +=1
                # print('---tile has no mask')
                continue
            if has_no_mask_pixels(tile):
                num_nomask_pixels +=1
                # print('---tile has no mask pixels')
                continue

            print("---tile Stats Before Normalization--------------------")
            for layer in tile.coords["layer"].values:
                # print(f"{layer}: Min={tile.sel(layer=layer).min().item()}, Max={tile.sel(layer=layer).max().item()}")
                # CALCULATE UNIQUE VALUES
                vals = np.unique(tile.sel(layer=layer).values)
                print(f"{layer}: num Unique values: {len(vals)}")

            # Normalize the tile
            # print('---normalizing tile')
            normalized_tile = normalize_inmemory_tile(tile)

            print("---Tile layers: ", tile.coords["layer"].values)

            # TODO  add to json
            tile_name = f"tile_{datacube_path.parent.name}_{x_start}_{y_start}.tif"
            # print('---tile_name= ', tile_name)

            dest_path = save_tiles_path / datacube_path.name / tile_name
            if not dest_path.parent.exists():
                os.makedirs(dest_path.parent, exist_ok=True)
                print('---created dir= ', dest_path.parent)

            # PRESAVE CHACK
            print("---Tile before saving---------------")
            for layer in normalized_tile.coords["layer"].values:
                vals = np.unique(tile.sel(layer=layer).values)
                print(f"{layer}: num Unique values: {len(vals)}")
            print('-----------------------------------')

            ######### SAVE TILE ############
            # Save layer names as metadata
            layer_names = list(normalized_tile.coords["layer"].values)
            layer_names = [str(name) for name in layer_names]
            print('---layer_names= ', layer_names)
            print('---normalized_tile layer names= ', normalized_tile.attrs["layer_names"]) 
            normalized_tile = normalized_tile.astype('float32')
            normalized_tile.rio.to_raster(dest_path, compress="deflate", nodata=np.nan)
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

            with rasterio.open(dest_path) as src:
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
    bands_to_check = ["dem", "slope", "vv", "vh", 'valid','mask']
    contains_nan = False  # Flag to track if any NaNs are found
    for band in bands_to_check:
        # Calculate the number of NaNs in the current band
        nan_count = np.isnan(tile.sel(layer=band)).sum().values
        if nan_count > 0:
            # print(f"Tile contains {nan_count} NaN values in {band}")
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

def has_no_mask_pixels(tile, ):
    #print("Values in 'mask' layer for tile:", tile.sel(layer='mask').values)
    return 1 not in tile.sel(layer='mask').values  # If valid data is found, return False
    

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

# GENERAL

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


# MAYBE NOT NEEDED

def nan_check(input):
    if np.isnan(input).any():
        print("----Warning: NaN values found in the data.")
        return False
    else:
        #print("----NO NANS FOUND")
        return True
    
def check_int16_range(dataarray):
    # TAKES A DATAARRAY NOT A DATASET
    #print("+++in small int16 range check fn+++")
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if (dataarray < int16_min).any() or (dataarray > int16_max).any():
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        # Calculate actual min and max values in the array
        actual_min = dataarray.min().item()
        actual_max = dataarray.max().item()
        
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        print(f"---Minimum value found: {actual_min}")
        print(f"---Maximum value found: {actual_max}")
        return False
    
    # else:
    #     print(f"---no exceedances int16.")

    # Optional: Replace NaN and Inf values if necessary
    # dataarray = dataarray.fillna(0)  # Replace NaN with 0 or another appropriate value
    # dataarray = dataarray.where(~np.isinf(dataarray), 0)  # Replace Inf with 0 or appropriate value

 
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
