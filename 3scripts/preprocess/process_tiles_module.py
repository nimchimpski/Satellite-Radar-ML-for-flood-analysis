import numpy as np
import rasterio
from pathlib import Path
import shutil
from tqdm import tqdm
import shutil
import random
import os
import xarray as xr

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
def has_enough_valid_pixels(file_path, threshold=0.8):
    """
    Check if the 'mask' layer has enough valid (flood, value=1) pixels.
    :return: True if the file meets the criteria, False otherwise.
    """
    with rasterio.open(file_path) as dataset:
        # Assuming the mask layer is identified by its name or index
        mask_layer = None
        analysis_extent_layer = None

        for idx, desc in enumerate(dataset.descriptions):
            if desc == "mask":  # Find the 'mask' layer by its description
                mask_layer = idx + 1  # Bands in rasterio are 1-based
            elif desc == "analysis_extent":
                analysis_extent_layer = idx + 1
        
        if mask_layer is None:
            print(f"No 'mask' layer found in {file_path}")
        if analysis_extent_layer is None:
            print(f"No 'analysis_extent' layer found in {file_path}")

        # Read the mask layer
        total_pixels = dataset.width * dataset.height
        mask_data = dataset.read(mask_layer)
        analysis_extent_data = dataset.read(analysis_extent_layer)

        # Calculate the proportion of pixels with value 1
        if (mask_data == 1).sum() == 0:
            print(f"No flood in maskfor: {file_path}")
            return False
        if ((analysis_extent_data == 1).sum()) / total_pixels < threshold:
            print(f"Insufficient extent pixels: {file_path}")
            return False
        return True

def select_tiles_and_split(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print('+++select_tiles_and_split')
    # Ensure the ratios sum to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Create destination folders
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"

    with open(dest_dir / "train.txt", "w") as traintxt,  open(dest_dir / "val.txt", "w") as valtxt,  open(dest_dir / "test.txt", "w") as testtxt:
        # Get a list of all files in the source directory
        files = list(source_dir.glob('*'))  # Modify '*' if you want a specific file extension

        # FILTER FILES BY VALID PIXELS
        filtered_files = [file for file in files if has_enough_valid_pixels(file)]
        # SHUFFLE FILES
        random.shuffle(filtered_files)  # Shuffle files for random split

        # Calculate split indices
        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * val_ratio)

        # Split files into train, val, and test sets
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

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
                print(f"Error copying {file}: {e}")

        for file in tqdm(test_files, desc="Copying test files"):
            try:
                shutil.copy(file, test_dir / file.name)
                testtxt.write(f"{file.name}\n")
            except Exception as e:
                print(f"Error copying {file}: {e}")

        # print(f"---Total files: {len(files)}")
        # print(f"---Train files: {len(train_files)}")
        # print(f'---Test files: {len(test_files)}')
        # print(f"---Validation files: {len(val_files)}")
        assert len(train_files) + len(val_files) + len(test_files) == len(files), "Files not split correctly"
        # with open("train.txt", "r") as tra, open("val.txt", "r") as val, open("test.txt", "r") as tes:
    with open(dest_dir / "train.txt", "r") as traintxt,  open(dest_dir / "val.txt", "r") as valtxt,  open(dest_dir / "test.txt", "r") as testtxt:

        # print('---len(train.txt)', len(train.readlines()))

        if len(traintxt.readlines()) != len(train_files):
            print('---train.txt not created successfully')
            print(f'---{traintxt.readlines()}')
        if len(valtxt.readlines()) != len(val_files):
            print('---val.txt not created successfully')
            print('---val.txt', valtxt.readlines())
        if len(testtxt.readlines()) != len(test_files):
            print('---test.txt not created successfully')
            print('---test.txt', testtxt.readlines())

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
