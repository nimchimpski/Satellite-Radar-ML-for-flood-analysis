import xarray as xr
import numpy as np
import rioxarray as rxr 

def dataset_type(da):
    if isinstance(da, xr.Dataset):
        print('---da is a dataset')
    else:
        print('---da is a dataarray')


def print_dataarray_info(da):
    print('-------------------------') 
    for layer in da.coords["layer"].values:
        layer_data = da.sel(layer=layer)
        print(f"---Layer '{layer}': Min={layer_data.min().item()}, Max={layer_data.max().item()}")
        print(f"---num unique vals = {len(np.unique(layer_data.values))}")
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

