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


def dataset_type(da):
    if isinstance(da, xr.Dataset):
        print('---da is a dataset')
    else:
        print('---da is a dataarray')

# CHECKS FOR INITIAL FOLDERS

def check_single_input_filetype(folder,  title, fsuffix):
    print(f"---Checking for {title} in {folder}")
    print(f"---Suffix: {fsuffix}")
    print(f"---Title: {title}")
    input_list = [i for i in folder.iterdir() if i.is_file() and i.suffix.lower() == fsuffix.lower() and title.lower() in i.name.lower()]

    if len(input_list) == 0:
        print(f"---No file with '{title}' found in {folder}")
        return None
    elif len(input_list) > 1:
        print(f"---Multiple images found in {folder}. Using the first one. Delete the rest!")
        return None
    return input_list[0]




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
