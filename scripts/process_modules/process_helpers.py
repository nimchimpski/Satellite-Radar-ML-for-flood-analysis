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


def print_tiff_info_TSX( image=None, mask=None):
    # print('---PRINT TIFF INFO---')
    if image:
        print(f'---in image = {image.name}') 

        with rasterio.open(image) as src:
            print(f'\n---CHECKING image = {image.name}') 
            data = src.read()
            # print layer names
            print(f"---Layer names: {src.descriptions}")        
            print(f"--- shape: {data.shape}, dtype: {data.dtype}, crs ={src.crs}")
            nan_check(data)
    if isinstance(mask, Path):
        with rasterio.open(mask) as src:
            # print(f'---mask type = {type(mask)}')
            # print(f'---CHECKING= {mask.name}')
            data = src.read(3)
            # print(f"--- shape: {data.shape}, dtype: {data.dtype}, crs ={src.crs}")
            unique_values = np.unique(data)
            if len(unique_values) > 1:
                print(f"%%%%%%%%%%%%%%%%%%%%%%Unique values in mask: {unique_values}")
            # nan_check(data)
    # elif isinstance(mask, DatasetReader):
    #     print
    #     print(f'---CHECKING= {mask.name}')
    #     data = mask.read()
    #     print(f"--- shape: {data.shape}, dtype: {data.dtype}, crs ={mask.rio.crs}")
    #     unique_values = np.unique(data)
    #     print(f"---Unique values in mask: {unique_values}")
    #     nan_check(data)
    # print('-----------------------')
    

def pad_tile(tile, expected_size=250, pad_value=0):
    current_x = tile.sizes["x"]
    current_y = tile.sizes["y"]

    # Calculate padding amounts
    pad_x = max(0, expected_size - current_x)
    pad_y = max(0, expected_size - current_y)

    if pad_x == 0 and pad_y == 0:
        # No padding needed
        return tile
