import os
import rasterio
import rioxarray as rxr # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from tqdm import tqdm
# import cv2
# import matplotlib.pyplot as plt
from pathlib import Path

VERSION = "UNOSAT-AI-v2"
TRACKING = "UNOSAT-AI-v1"


# TODO add rtc function
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

def check_layers(layers, layer_names):
    '''
    checks the layers and prints out some info
    '''
    print('\n+++in check_layers fn+++++++++++++++++++++++++')
    # Assuming you have a list of Dask arrays, each representing a layer
    
    for i, layer in enumerate(layers):
        print(f'---layer name = {layer_names[i]}')
        print(f'---layer type = {type(layer)}')
        print(f"Layer {i+1}:")

        # Print the shape of the layer
        print(f"  Shape: {layer.shape}")

        # Print the data type of the layer
        print(f"  Data Type: {layer.dtype}")

        # Assuming the array has x and y coordinates in the `.coords` attribute (like in xarray)
        # You can access and print the coordinates if it is an xarray DataArray or a similar structure
        if hasattr(layer, 'coords'):  # If using Dask with xarray-like data
            print(f"  X Coordinates: {layer.coords['x']}")
            print(f"  Y Coordinates: {layer.coords['y']}")
        else:
            print("  No coordinate information available.")

        missing_values = np.isnan(layer.values).sum()
        print(f"--- Missing Values (NaN): {missing_values}")

        print("\n")  # Separate the output for each layer

def create_vv_and_vh_tifs(event):
    '''
    will delete the original image after creating the vv and vh tifs
    '''
    print('+++in create_vv_and_vh_tifs fn')
    # List files in the directory and print them for debugging
    print('length files in event' , len(list(event.iterdir())))
    for file in event.iterdir():
        # Check if the event directory exists and print the path
        if not event.exists() or not event.is_dir():
            print(f"---Error: Event directory {event} does not exist or is not a directory")
            return

        print(f'---looking at= {file.name}')
        if 'img.tif' in file.name:
            print(f'---found image file= {file.name}')
            # Open the multi-band TIFF
            with rasterio.open(file) as src:
                # Read the vv (first band) and vh (second band)
                vv_band = src.read(1)  # Band 1 (vv)
                vh_band = src.read(2)  # Band 2 (vh)

                # Define metadata for saving new files
                meta = src.meta

                # Update meta to reflect the single band output
                meta.update(count=1)

                # Save the vv band as a separate TIFF
                vv_newname = file.name.rsplit('_', 1)[0]+'_vv.tif'
                print('---vv_newname= ',vv_newname)
                with rasterio.open(file.parent / vv_newname, 'w', **meta) as destination:
                    destination.write(vv_band, 1)  # Write band 1 (vv)

                # Save the vh band as a separate TIFF
                vh_newname = file.name.rsplit('_', 1)[0]+'_vh.tif'
                with rasterio.open(file.parent / vh_newname, 'w', **meta) as destination:
                    destination.write(vh_band, 1)  # Write band 2 (vh)
            file.unlink()  # Delete the original image file  

  
                # delete original image using unlink() method
    print('---finished create_vv_and_vh_tifs fn')

def matchresolutions(event):
    """Match the resolution and dimensions of the target raster to the reference raster."""
    print('+++in matchresolutions fn')
    reference_path = None
    
    # Find the reference file (vv.tif)
    for file in event.iterdir():
        if 'vv.tif' in file.name:
            reference_path = file
            print(f'---vv reference_path= {reference_path}')
            break  # Stop after finding the reference path
    
    if reference_path:
        # Loop through files and reproject all except the vv and vh files
        for file in event.iterdir():
            if 'vv.tif' not in file.name and 'vh.tif' not in file.name:
                print(f'---found file to reproject = {file}')
                # print(f'---target reference_path= {reference_path.name}')
                
                # Open the reference and target rasters inside `with` blocks
                with rxr.open_rasterio(reference_path) as reference_layer, rxr.open_rasterio(file) as target_layer:
                    # Reproject the target raster to match the reference raster
                    reprojected_layer = target_layer.rio.reproject_match(reference_layer)

                # Now that both files are closed, save the reprojected layer directly
                reprojected_layer.rio.to_raster(file)
                print(f'---Resampled raster saved to: {file}')

    else:
        print('---Error: reference vv.tif not found.')





def make_eventcube(data_root, event, datas):
    '''
    Loop through each dataset  and load into xarray datacube (stacked on a new dimension)
    'Event' is a single folder conatining flood event data    
    'satck' is an Xarray DataArray 
    '''   
    print(f"++++ in Eventcube fn")      
    layers = []
    layer_names = []
    # CREATE THE DATACUBE
    for tif_file, band_name in datas.items():
        try:
            print(f"\n---Loading {tif_file}-------------------------------------------")

            # Use Dask chunking for large .tif files
            stack = rxr.open_rasterio(Path(data_root, event, tif_file), chunks={'x': 1024, 'y': 1024})
            # Check if stack is created successfully
            if stack is None:
                print(f"---Error: Stack for {tif_file} is None")
            # else:
            #     # Print basic info about the stack
            #     print(f"---Successfully loaded stack for {tif_file}")
                # print(f"---Stack type: {type(stack)}")  # Should be an xarray.DataArray
                # print(f"---Stack dimensions: {stack.dims}")
                # print(f"---Stack shape: {stack.shape}")  # Check the shape of the array
                # print(f"---Stack data type: {stack.dtype}")  # Check the data type of the array
                # print(f"---Stack chunk sizes: {stack.chunks}")
                # # Print the first few coordinate values for x and y
                # print(f"---X Coordinates: {stack.coords['x'].values[:5]}")  # First 5 x-coordinates
                # print(f"---Y Coordinates: {stack.coords['y'].values[:5]}")  # First 5 y-coordinates
       
        except Exception as e:
            print(f"---Error loading {tif_file}: {e}")
            continue  # Skip this file if there is an issue loading
        # Handle multi-band image data (e.g., VV, VH bands)
    
        try:
            print(f"---Processing single-band layer: {band_name}")
            layers.append(stack)
            layer_names.append(band_name)
            print(f"---Successfully processed layer with band name: {band_name}")
        except Exception as e:
            print(f"---Error creating layers for {tif_file}: {e}")
            continue
    print(f'---finished {event.name}\n'  )   
    print('---length layers= ',len(layers), '\n')         
    print(f'---layer_names= {layer_names}\n')
    if layers:
        
        # Convert layers to float32, except for 'mask' and 'valid' layers

        check_layers(layers, layer_names)
        print('---layers checked ok')
        # Concatenate all layers along a new dimension called 'layer' or 'band'
        eventcube = xr.concat(layers, dim='layer').astype('int16')
        print('---eventcube made ok')

        # Assign the layer names (e.g., ['vv', 'vh', 'dem', 'slope']) to the 'layer' dimension
        eventcube = eventcube.assign_coords(layer=layer_names)
        print('---layer names assigned ok')
        # Rechunk the final datacube for optimal performance
        eventcube = eventcube.chunk({'x': 1024, 'y': 1024, 'layer': 1})
        print('---Rechunked datacube')

        # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
        if 'band' in eventcube.dims and eventcube.sizes['band'] == 1:
            eventcube = eventcube.squeeze('band')
            print('---Squeezed out single-band dimension')

        return eventcube
    else:
        print('---No layers found')
        return None 
    

def arsemake_eventcube(data_root, event, datas):
    '''
    Loop through each dataset to load into xarray datacube, ensuring layers retain their dtype
    '''
    print(f"++++ in new Eventcube fn")   
    float_layers = {}
    uint8_layers = {}
    
    for tif_file, band_name in datas.items():
        try:
            if 'mask' in tif_file or 'valid' in tif_file:
                # Load as uint8
                stack = rxr.open_rasterio(Path(data_root, event, tif_file), chunks={'x': 1024, 'y': 1024}).astype('uint8')
                uint8_layers[band_name] = stack
                print(f"---*********Loaded {tif_file} as uint8")
                # check this is 8bit
                print(f"---Data type for {tif_file}: {stack.dtype}")
            else:
                # Load as float32 for other layers
                stack = rxr.open_rasterio(Path(data_root, event, tif_file), chunks={'x': 1024, 'y': 1024}).astype('float32')
                float_layers[band_name] = stack
        
        except Exception as e:
            print(f"Error loading {tif_file}: {e}")
            continue
    
    # Convert float layers dict to Dataset
    if float_layers:
        float_eventcube = xr.Dataset(float_layers)
    else:
        float_eventcube = None
    
    # Convert uint8 layers dict to Dataset
    if uint8_layers:
        uint8_eventcube = xr.Dataset(uint8_layers)
        # Check the data type of the uint8 layers
        for layer_name in uint8_eventcube.data_vars:
            print(f"************Data type for {layer_name}: {uint8_eventcube[layer_name].dtype}")
    else:
        uint8_eventcube = None
    
    # Merge float32 and uint8 eventcubes
    if float_eventcube is not None and uint8_eventcube is not None:
        eventcube = xr.merge([float_eventcube, uint8_eventcube], compat='override')
        # Check the data type of the merged eventcube
        for layer_name in eventcube.data_vars:
            print(f"***************Data type for {layer_name}: {eventcube[layer_name].dtype}")
    elif float_eventcube is not None:
        eventcube = float_eventcube
    else:
        eventcube = uint8_eventcube
    
    return eventcube


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
        elif 'vv.tif' in file.name:
            # print(f'---image vv file found {file}') 
            datas[file.name] = 'vv'
        elif 'vh.tif' in file.name:
            # print(f'---image vh file found {file}')   
            datas[file.name] = 'vh'
    print('---datas ',datas)
    return datas

def datacube_check(datacube):
    """
    Perform a quick health check on the final datacube to ensure its consistency.
    
    Parameters:
    - datacube (xarray.DataArray or xarray.Dataset): The final datacube to check.
    
    Prints out:
    - General info about the datacube (dimensions, chunking)
    - Data type of each layer
    - Range of coordinates (x, y)
    - Any NaN or missing values
    """
    print("+++ Datacube Health Check +++")
    
    # General info about the datacube
    print("\n--- Datacube Dimensions ---")
    print(datacube.dims)
    
    print("\n--- Datacube Shape ---")
    print(datacube.shape)
    
    print("\n--- Datacube Chunk Sizes ---")
    if datacube.chunks:
        print(datacube.chunks)
    else:
        print("No chunking detected.")
    
    print("\n--- Coordinate Ranges ---")
    print(f"X Range: {datacube.coords['x'].min().values} to {datacube.coords['x'].max().values}")
    print(f"Y Range: {datacube.coords['y'].min().values} to {datacube.coords['y'].max().values}")
    
    print("\n--- Data Types ---")
    if isinstance(datacube, xr.Dataset):
        for var_name, var_data in datacube.data_vars.items():
            print(f"Layer {var_name}: {var_data.dtype}")
    elif isinstance(datacube, xr.DataArray):
        print(f"Data type: {datacube.dtype}")
    
    print("\n--- Checking for NaN or Missing Values ---")
    if isinstance(datacube, xr.Dataset):
        for var_name in datacube.data_vars:
            missing_values = datacube[var_name].isnull().sum().values
            print(f"Layer {var_name}: {missing_values} missing values")
    elif isinstance(datacube, xr.DataArray):
        missing_values = datacube.isnull().sum().values
        print(f"DataArray: {missing_values} missing values")
    
    print("\n--- Checking Memory Usage ---")
    memory_usage = datacube.nbytes / (1024**3)  # Convert to GB
    print(f"Datacube Memory Usage: {memory_usage:.2f} GB")
    
    print("\n+++ Datacube Health Check Completed +++")

def main():
    all_eventcubes = []  # List to hold datacubes for each event
    event_names = []  # List to hold event names
    # TODO add RTC functionality
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    for event in data_root.iterdir():
        # prepare the images
        if event.is_dir():
            print(f"---Preprocessing event: {event.name}")
            # split image into seperate vv and vh tiffs
            create_vv_and_vh_tifs(event)
            # reproject everything to match higher resolution
            matchresolutions(event)     
        # Get the datas info from the folder
        datas = make_datas(event)
        # Create the eventcube
        eventcube = make_eventcube(data_root, event, datas)
        print('---eventcube made for= ',event.name, '\n')
        all_eventcubes.append(eventcube)
        event_names.append(event.name)
    print('---finished all events\n')
    print('---event_names= ',event_names)
    print('---############    all_eventcubes= ###########\n',all_eventcubes)
    # combine all eventcubes into a single datacube
    datacube = xr.concat(all_eventcubes, dim='event').astype('int16')
    datacube = datacube.assign_coords(event=event_names)
    print('--->>>>>>>>>final datacube check::: ')
    datacube_check(datacube)
if __name__ == "__main__":
    main()

    # TODO add checks for failed eventdatacubes