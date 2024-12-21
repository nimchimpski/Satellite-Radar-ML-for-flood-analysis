import xarray as xr
import numpy as np
import rioxarray as rxr 
import rasterio
from pathlib import Path
from rasterio.io import DatasetReader

def dataset_type(da):
    if isinstance(da, xr.Dataset):
        print('---da is a dataset')
    else:
        print('---da is a dataarray')


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
                print(f"---Band {name}: Min={min}, Max={max}")

                print(f"---CRS: {src.crs}")
            # print layer names




# SUBFUNCS FOR MULTI BAND TILES/TIFS

def get_band_name(band, src):
    return src.descriptions[band - 1].lower() if src.descriptions[band - 1] else None


def num_band_vals(band_data):
    return len(np.unique(band_data))

def min_max_vals(band_data): # IF UNIQUE VALS NOT 0 OR 1 - FLAG IT
    return np.min(band_data), np.max(band_data)

def datatype_check(band_data):
    return band_data.dtype  
