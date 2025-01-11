import xarray as xr
import numpy as np
import rioxarray as rxr 
import rasterio
from pathlib import Path
from rasterio.io import DatasetReader
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import fiona
import json
import random
import sys
from tqdm import tqdm


# CHECKS FOR INITIAL FOLDERS

def check_single_input_filetype(folder,  title, fsuffix):
    # print(f"---Checking for {title} in {folder}")
    # print(f"---Suffix: {fsuffix}")
    # print(f"---Title: {title}")
    input_list = [i for i in folder.iterdir() if i.is_file() and i.suffix.lower() == fsuffix.lower() and title.lower() in i.name.lower()]

    if len(input_list) == 0:
        print(f"---No file with '{title}' found in {folder}")
        return None
    elif len(input_list) > 1:
        print(f"---Multiple images found in {folder}. Using the first one. Delete the rest!")
        return None
    return input_list[0]

def path_not_exists(input_path):
    if not  input_path.exists():
        print(f"---no {input_path.name} found in {input_path}")
        return True

# NORMALISING
def rescale_image_minmax(image, min, max, output_path):
    """
    Rescales the pixel values of an image to a new range.
    
    Parameters:
        image (str or Path): Path to the input image.
        min (float): Minimum pixel value.
        max (float): Maximum pixel value.
        output_path (str or Path): File path to save the normalized image.
    """
    output_path = Path(output_path)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the image
    with rasterio.open(image) as src:
        data = src.read()

        # Normalize the pixel values
        data = (data - min) / (max - min)

        # Write the normalized image
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            width=src.width,
            height=src.height,
            count=src.count,
            crs=src.crs,
            transform=src.transform,
            dtype=data.dtype
        ) as dst:
            dst.write(data)

    print(f"Normalized image saved to {output_path}")

def compute_dataset_minmax(dataset, band_to_read=1):
    """
    Computes the global minimum and maximum pixel values for a dataset.
    
    Parameters:
        dataset_dir (str or Path): Directory containing all input images.
    
    Returns:
        global_min (float): Global minimum pixel value.
        global_max (float): Global maximum pixel value.
    """
    global_min = float('inf')
    global_max = float('-inf')
    
    # Iterate through all TIFF files
    for event in dataset.iterdir(): # ITER EVENT
        print(f'---event= {event.name}')    
        if event.is_dir():
            images = list(event.rglob('*IMAGE_*.tif') )
            print(f'---num images= {len(images)}')
            if len(images) != 1:
                raise (f"---Error: {event.name} contains {len(images)} images. Skipping...")
            image = images[0]

            # print(f"---Processing {image}")
            try:
                lmin, lmax = compute_image_minmax(image, band_to_read)
                global_min = int(min(global_min, lmin))
                global_max = int(max(global_max, lmax))
                print(f'---global_min={global_min}, global_max={global_max}')
            except Exception as e:
                print(f"Error processing {image}: {e}")
                continue

    print(f"Global Min: {global_min}, Global Max: {global_max}")
    return global_min, global_max

def check_loc_max_and_rescale(image, glob_max, output_path, band_to_read=1):
    with rasterio.open(image)as src:
        data = src.read(band_to_read)  # Read the first band
        locmin, locmax = int(data.min()), int(data.max())
        if locmax < 0.8*glob_max:
            data = rescale_image_minmax(data, glob_max, locmax)  
            print(f"---Rescaled {image.name} from {locmax} to {glob_max}")      
            print(f'---min { int(data.min())}, max { int(data.max())}')
            # Write the normalized image
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                width=src.width,
                height=src.height,
                count=src.count,
                crs=src.crs,
                transform=src.transform,
                dtype=data.dtype
            ) as dst:
                dst.write(data)


def write_minmax_to_json(min, max, output_path):
    """
    Writes min and max values for each variable to a JSON file.

    Args:
        min_max_values (dict): Dictionary containing min and max values for each variable.
                               Example: {"SAR_HH": {"min": 0.0, "max": 260.0}, "DEM": {"min": -10.0, "max": 3000.0}}
        output_path (str or Path): File path to save the JSON file.
    """
    print(f'---minmaxvalsdict= {min, max}')
    output_path = Path(output_path)

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the dictionary to the JSON file
    with open(output_path, 'w') as json_file:
        json.dump({'hh': {'min': min, 'max' : max}}, json_file, indent=4)
    
    print(f"Min and max values saved to {output_path}")


def read_minmax_from_json(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    return data.get("hh", {})


def compute_image_minmax(image, band_to_read=1):
    with rasterio.open(image) as src:
        # Read the data as a NumPy array
        data = src.read(band_to_read)  # Read the first band
        min, max = int(data.min()), int(data.max())
        print(f"---{image.name}: Min: {data.min()}, Max: {data.max()}")
        # get image size in pixels
        print(f"---: Shape: {data.shape}")
    return min, max

def rescale_image_minmax( data, glob_max, loc_max):
        data = data * glob_max / loc_max
        return data

# FUNCTIONAL
def read_raster(image_path, band_to_read=1):
    """Reads a raster band and returns the data, metadata, and transform."""
    with rasterio.open(image_path) as src:
        data = src.read(band_to_read)
        metadata = src.meta.copy()
    return data, metadata

def rescale_image_minmax(data, glob_max, loc_max):
    """Rescales the image data to align `loc_max` with `glob_max`."""
    scale_factor = glob_max / loc_max
    return data * scale_factor, scale_factor

def check_and_rescale(data, metadata, glob_max, threshold=0.8):
    """Checks the local max and rescales the data if below a threshold."""
    loc_min, loc_max = data.min(), data.max()
    print(f"---Local min: {loc_min}, Local max: {loc_max}")
    
    if loc_max < threshold * glob_max:
        data, scale_factor = rescale_image_minmax(data, glob_max, loc_max)
        print(f"---Rescaled from {loc_max} to {glob_max}")
    else:
        print("---No rescaling needed.")
        scale_factor = 1.0
    
    return data, metadata, scale_factor

def write_raster(output_path, data, metadata):
    """Writes a raster dataset to the specified output path."""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=metadata['crs'],
        transform=metadata['transform']
    ) as dst:
        dst.write(data, 1)

def process_raster_minmax(image_path, output_path, glob_max, threshold=0.8):
    """Functional pipeline for checking and rescaling raster data."""
    data, metadata = read_raster(image_path)
    data, metadata, scale_factor = check_and_rescale(data, metadata, glob_max, threshold)
    write_raster(output_path, data, metadata)
    return scale_factor

# DATAARRAY TESTS

def dataset_type(da):
    if isinstance(da, xr.Dataset):
        print('---da is a dataset')
    elif isinstance(da, xr.DataArray):
        print('---da is a dataarray')
    else:
        print('---da is not a dataset or dataarray')



def open_dataarray(nc):
    da =xr.open_dataarray(nc)
    return da

def print_dataarray_info(da):
    print('-----------PRINT DATARAY INFO--------------') 
    for layer in da.coords["layer"].values:
        layer_data = da.sel(layer=layer)
        print(f"---Layer '{layer}': Min={layer_data.min().item()}, Max={layer_data.max().item()}")
        print(f"---num unique vals = {len(np.unique(layer_data.values))}")
        if len(np.unique(layer_data.values)) < 4:
            print(f"---unique vals = {np.unique(layer_data.values)}")
        print(f'---Layer crs={layer_data.rio.crs}')   
        print('-------------------------') 

def check_dataarray_list(dataarrays, layer_names):
    for i, da in enumerate(dataarrays):
        if not hasattr(da, 'rio'):
            print(f"---Error: {layer_names[i]} lacks rioxarray accessors. Reinitializing...")
        else:
            print('---has attr')
        print('---type da= ',type(da))  
        print(f"---Layer {i} name: {layer_names[i]}")
        print(f"---Shape: {da.shape}, CRS: {da.rio.crs}, Resolution: {da.rio.resolution()}, Bounds: {da.rio.bounds()}")
        if da.rio.crs != dataarrays[0].rio.crs:
            print(f"---Mismatch in CRS for {layer_names[i]}")
        if da.rio.resolution() != dataarrays[0].rio.resolution():
            print(f"---Mismatch in Resolution for {layer_names[i]}")
        dataarrays[i] = da.astype('float32')
        # chack the datatype
        print(f"---Data Type: {da.dtype}")


def nan_check(nparray):
    if np.isnan(nparray).any():
        print("----Warning: NaN values found in the data.")
        return False
    else:
        print("----NO NANS FOUND")
        return True


def pad_tile(tile, expected_size=250, pad_value=0):
    current_x = tile.sizes["x"]
    current_y = tile.sizes["y"]

    # Calculate padding amounts
    pad_x = max(0, expected_size - current_x)
    pad_y = max(0, expected_size - current_y)

    if pad_x == 0 and pad_y == 0:
        # No padding needed
        return tile

# CHECKS FOR TILES (MULTIBAND TIFS)

def print_tiff_info_TSX( image):
    print(f'+++ PRINT TIFF INFO---{image.name}')
    if image:
        print(f'---in image = {image.name}') 

        with rasterio.open(image) as src:
            data = src.read()
            nan_check(data)
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                min, max = min_max_vals(band_data)
                name = get_band_name(i, src)
                numvals =  num_band_vals(band_data)
                print(f"---Band {name}: Min={min}, Max={max}")
                print(f"---num unique vals = {numvals}")
                print(f"---CRS: {src.crs}")
        
def check_single_tile(tile):
    with rasterio.open(tile) as src:
        # print('---tile:', tile.name)
        # Read all datasdsdfsdfsdvsdvs
        data = src.read()
        # LOOP THRU BANDS
        for band in range(1, src.count + 1):
            band_data = data[band - 1]
            name = get_band_name(band, src)
            print(f'\n---{band}={name}')
            numvals =  num_band_vals(band_data)
            if name in ['mask', 'extent']:
                # CHECK NUM UNIQUE VALS
                if numvals >2:
                    print('---not 2 vals , ', numvals)
                    return
                  # CHECK VALS ARE 0 OR 1
                min, max = min_max_vals(band_data)
                if round(min) not in [0, 1] or round(max) not in [0, 1]:
                    print(f'---min={min}, max={min}')
                    pass
            else:
                # print(f'--num_band_vals={numvals}')
                # CHECK MIN MAX INSIDE 0 AND 1 - NORMALIZED
                min, max = min_max_vals(band_data)
                if min == max:
                    print(f'---uniform values in {name} band: {min}, {max}')
                if min < 0 or max > 1:
                    print(f'---out of range values in {name} band: {min}, {max}')
                    raise ValueError(f'---out of range values in {name} band: {min}, {max}')


def rasterize_kml_rasterio(kml_path, output_path, pixel_size=0.0001, burn_value=1):
    # Convert KML to GeoJSON using Fiona
    print(f"+++++Rasterizing extent from {kml_path} to {output_path}")
    with fiona.open(kml_path, 'r') as src:
        geometries = [shape(feature['geometry']) for feature in src]

    # Get extent
    xmin, ymin, xmax, ymax = src.bounds

    # Define raster metadata
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, 
    int((xmax - xmin) / pixel_size), int((ymax - ymin) / pixel_size))
    height, width = int((ymax - ymin) / pixel_size), int((xmax - xmin) / pixel_size)

    # Create and save the raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        raster = rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=burn_value,
            dtype=rasterio.uint8
        )
        dst.write(raster, 1)

    print(f"Rasterized extent saved to {output_path}")





# SU    BFUNCS FOR MULTD TILES/TIFS

def get_band_name(band, src):
    return src.descriptions[band - 1].lower() if src.descriptions[band - 1] else None


def num_band_vals(band_data):
    return len(np.unique(band_data))

def min_max_vals(band_data): # IF UNIQUE VALS NOT 0 OR 1 - FLAG IT
    return np.min(band_data), np.max(band_data)

def datatype_check(band_data):
    return band_data.dtype  


def handle_interrupt(signal, frame):
    '''
    usage: signal.signal(signal.SIGINT, handle_interrupt)
    '''
    print("Interrupt received! Cleaning up...")
    # Add any necessary cleanup code here (e.g., saving model checkpoints)
    sys.exit(0)

def calc_ratio(tiles):
    flooded_count = 0
    non_flooded_count = 0
    for tile in tqdm(tiles.iterdir(), total=len(list(tiles.iterdir()))):
        if tile.suffix != ".tif":
            continue
        # print(f"---Processing {tile.name}")
        with rasterio.open(tile) as src:
            data = src.read(3)
            flooded_count += np.sum(data == 1)
            non_flooded_count += np.sum(data == 0)
            # print(f'---flooded_count: {flooded_count}')


    # Calculate class ratio
    total_pixels = flooded_count + non_flooded_count
    class_ratio = flooded_count / total_pixels
    # print(f'---event: {event.name}')
    print(f"{tile.parent.name} Ratio: {class_ratio:.2f}")
    return class_ratio