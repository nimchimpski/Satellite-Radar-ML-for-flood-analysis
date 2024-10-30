import os
import rasterio
import rioxarray as rxr # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from tqdm import tqdm
from pathlib import Path
import netCDF4 as nc
from osgeo import gdal
from check_int16_exceedance import check_int16_exceedance
from modules.helpers import nan_check   
# from tile_datacube import tile_datacube c
'''
- will overwrite existing .nc files in the event folders.
- expects each event folder to have the following files:
    elevation.tif
    slope.tif
    msk.tif
    valid.tif

TODO check process in place to organise the dataroot / input files such as they are correct for this script.
TODO THERE IS A PERSISTANT HIDDEN PROBLEM WITH WRONG VALUES AFTER CASTING.
- 
'''


# TODO add rtc function


def fill_nodata_with_zero(input_file):
    print('+++in fill_nodata_with_zero fn')
    input_file = str(input_file)
    # Open the input raster file
    dataset = gdal.Open(input_file, gdal.GA_Update)
    band = dataset.GetRasterBand(1)

    # Get the nodata value (if it exists)
    nodata_value = band.GetNoDataValue()

    # Read the raster as a NumPy array
    data = band.ReadAsArray()

    # Replace nodata values with 0
    if nodata_value is not None:
        print('---found nans')
        data[data == nodata_value] = 0
    
    # Write the modified array back to the raster
    band.WriteArray(data)
    
    # Flush and close the file
    band.FlushCache()
    dataset = None  # Close the file

def filter_allones(tiled_datacube):
    # Example: After tiling, you can iterate through tiles to verify them
    for i, tile in enumerate(tiled_datacube):
        if np.all(tile == 1):
            print(f"Warning: Tile {i} has all values as 1. Check mask or data integrity.")

def filter_noanalysis(tile):
    '''
    Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
    '''
    if 0 in tile.sel(band='analysis_extent').values:
        return False
    return True

def tile_datacube(datacube, tile_size_x=256, tile_size_y=256):
    '''
    Tile the datacube into smaller chunks (tiles) for efficient processing.
    The tiles will have size tile_size_x x tile_size_y.
    '''
    print(f"+++ Tiling Datacube +++")
    if not datacube.rio.crs:
        raise ValueError("The datacube is missing a CRS. Please assign one before tiling.")

    
    # Rechunk the datacube with the specified tile sizes
    tiled_datacube = datacube.chunk({'x': tile_size_x, 'y': tile_size_y, 'layer': 1})

    # Example: Write each tile to disk after chunking if necessary
    for tile in tiled_datacube:
        tile_path = data_root / f"tile_{i}.tif"
        tile.rio.to_raster(tile_path)


    # Check if tiling has been done correctly
    print(f"---Tiled Datacube Chunk Sizes: {tiled_datacube.chunks}")
    
    return tiled_datacube

def check_layers(layers, layer_names):
    '''
    checks the layers and prints out some info
    '''
    print('\n+++in check_layers fn+++++++++++++++++++++++++')
    # Assuming you have a list of Dask arrays, each representing a layer
    
    for i, layer in enumerate(layers):
        print(f'---layer name = {layer_names[i]}')
        print(f'---layer type = {type(layer)}')
        print(f"---Layer {i+1}:")

        # Print the shape of the layer
        print(f"---Shape: {layer.shape}")

        # Print the data type of the layer
        print(f"---Data Type: {layer.dtype}")

        # Assuming the array has x and y coordinates in the `.coords` attribute (like in xarray)
        # You can access and print the coordinates if it is an xarray DataArray or a similar structure
        if hasattr(layer, 'coords'):  # If using Dask with xarray-like data
            print(f"---X Coordinates: {layer.coords['x']}")
            print(f"---Y Coordinates: {layer.coords['y']}")
        else:
            print("---No coordinate information available.")

        # Check for NaN or Inf values in the layer
        check_invalid_values(layer)

        print("\n")  # Separate the output for each layer

def create_vv_and_vh_tifs(file):
    '''
    will delete the original image after creating the vv and vh tifs
    '''
    print('+++in create_vv_and_vh_tifs fn')

    # print(f'---looking at= {file.name}')
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


def match_resolutions_with_check(event):
    """
    Match the resolution and dimensions of the target raster to the reference raster
    only if they differ.
    """
    print('+++++++in match_resolutions_with_check fn')

    # Find the reference file (vv.tif)
    reference_path = None
    for file in event.iterdir():
        if 'vv.tif' in file.name:
            reference_path = file
            break
    
    if not reference_path:
        print('--- No reference vv.tif file found.')
        return
    
    # Open the reference layer to get its resolution
    with rxr.open_rasterio(reference_path) as reference_layer:
        reference_resolution = reference_layer.rio.resolution()
        print(f'--- Reference file {reference_path.name} resolution: {reference_resolution}')
    
    # Loop through other files to check and reproject if necessary
    for file in event.iterdir():
        if file.is_file() and file.name not in ['vv.tif', 'vh.tif'] and file.suffix != '.nc':
            print(f'--- Found file to reproject = {file.name}')

            # Open the target layer to compare resolutions
            with rxr.open_rasterio(file) as target_layer:
                target_resolution = target_layer.rio.resolution()
                print(f'--- Target file {file.name} resolution: {target_resolution}')

                # Compare resolutions with a small tolerance
                if abs(reference_resolution[0] - target_resolution[0]) < 1e-10 and \
                   abs(reference_resolution[1] - target_resolution[1]) < 1e-10:
                    print(f"--- Skipping {file.name} (resolution already matches)")
                    continue

                # Reproject target file to match reference
                reprojected_layer = target_layer.rio.reproject_match(reference_layer)
            # Save reprojected file (once target file is closed)
            reprojected_layer.rio.to_raster(file)
            print(f'--- Resampled raster saved to: {file}')

def make_eventcube(data_root, event, datas):
    '''
    MAKES AN XARRAY 'DATASET' (!!!)
    THIS MEANS 1ST DIMENSION IS A VARIABLE NAME. DEFAULT = '--xarray_dataarray_variable--'
    THIS MEANS FURTHER VARIABLES CAN BE ADDED - EG TIME SERIES DATA FOR SAME AREA
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
            # stack = rxr.open_rasterio(Path(data_root, event, tif_file), chunks={'x': 1024, 'y': 1024})
            stack = rxr.open_rasterio(Path(data_root, event, tif_file))
            # Check if stack is created successfully
            if stack is None:
                print(f"---Error: Stack for {tif_file} is None")
            else:
                # Print basic info about the stack
                nan_check(stack)

                # print(f"---Successfully loaded stack for {tif_file}")
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
        layers = [layer.broadcast_like(layers[0]) for layer in layers]

        check_layers(layers, layer_names)
        print('---layers checked ok')
        # Concatenate all layers along a new dimension called 'layer' or 'band'
        eventcube = xr.concat(layers, dim='layer').astype('int16')
        print('---eventcube made ok')

        check_invalid_values(eventcube)

        # Assign the layer names (e.g., ['vv', 'vh', 'dem', 'slope']) to the 'layer' dimension
        eventcube = eventcube.assign_coords(layer=layer_names)
        print('---layer names assigned ok')
        # Rechunk the final datacube for optimal performance
        eventcube = eventcube.chunk({'x': 1024, 'y': 1024, 'layer': 1})
        print('---Rechunked datacube')

        check_invalid_values(eventcube) 

        # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
        if 'band' in eventcube.dims and eventcube.sizes['band'] == 1:
            eventcube = eventcube.squeeze('band')
            print('---Squeezed out single-band dimension')

        return eventcube
    else:
        print('---No layers found')
        return None 
    
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
            datas[file.name] = 'valid'
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
        print("---No chunking detected.")
    
    print("\n--- Coordinate Ranges ---")
    print(f"---X Range: {datacube.coords['x'].min().values} to {datacube.coords['x'].max().values}")
    print(f"---Y Range: {datacube.coords['y'].min().values} to {datacube.coords['y'].max().values}")
    
    print("\n--- Data Types ---")
    if isinstance(datacube, xr.Dataset):
        for var_name, var_data in datacube.data_vars.items():
            print(f"---Layer {var_name}: {var_data.dtype}")
    elif isinstance(datacube, xr.DataArray):
        print(f"---Data type: {datacube.dtype}")
    
    print("\n--- Checking for NaN or Missing Values ---")
    if isinstance(datacube, xr.Dataset):
        for var_name in datacube.data_vars:
            missing_values = datacube[var_name].isnull().sum().values
            print(f"---Layer {var_name}: {missing_values} missing values")
    elif isinstance(datacube, xr.DataArray):
        missing_values = datacube.isnull().sum().values
        print(f"---DataArray: {missing_values} missing values")
    
    print("\n--- Checking Memory Usage ---")
    memory_usage = datacube.nbytes / (1024**3)  # Convert to GB
    print(f"---Datacube Memory Usage: {memory_usage:.2f} GB")

    # check invlaid values
    check_invalid_values(datacube)

    
    print("\n+++ Datacube Health Check Completed +++")

def check_invalid_values(dataarray):
    print("\n+++ Checking for Invalid Values +++")
    
    # Check for NaN values
    nan_check(dataarray)

    
    # Check for infinite values
    if np.isinf(dataarray).any():
        print("---Warning: Infinite values found in the data.")

    # Check for values outside the int16 range
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if (dataarray < int16_min).any() or (dataarray > int16_max).any():
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")

    # Optional: Replace NaN and Inf values if necessary
    # dataarray = dataarray.fillna(0)  # Replace NaN with 0 or another appropriate value
    # dataarray = dataarray.where(~np.isinf(dataarray), 0)  # Replace Inf with 0 or appropriate value

    return dataarray



def main():

    VERSION = "alex"
    TRACKING = "id_alex"

    event_names = []  # List to hold event names

    # TODO add RTC functionality
    data_root = Path(r"\\cerndata100\AI_files\Users\AI_flood_service\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")

    # LOOP THE FOLDERS BY  COUNTRIES, SOME REPEATED WITH DIF NUMBERED SUFFIX 
    for event in data_root.iterdir(): # CREATES ITERABLE FO FILE PATHS 

        # IGNORE .NC FILES
        if event.suffix == '.nc' or event.name in ['tiles']:
            # print(f"Skipping file: {event.name}")
            continue

        if event.is_dir() and any(event.iterdir()):
            print(f">>>Preprocessing event: {event.name}")

            # LOOP FILES IN EACH FOLDER
            for file in event.iterdir():
                if not file.is_dir() or file.suffix in {'.xml', '.json', '.nc'} or '_S2_' in file.name or 'epsg4326' not in file.name:
                    # print(f"Skipping file: {file.name}")
                    continue

                # DO WORK ON THE TIFS

                if file.name.endswith('slope.tif'):
                    print('>>>----filling nans in slope.tiff----------------')
                    fill_nodata_with_zero(file)
                    print('>>>cheking nans in filled slope.tiff')

                    # EXTRACT THE VV AND VH BANDS FROM THE IMAGE FILE
                    create_vv_and_vh_tifs(file)

            # reproject everything to match higher resolution - needs to loop through whole event folder to find the vv.tif as reference
            print('>>>will now match resolutions')
            match_resolutions_with_check(event)     

        # Get the datas info from the folder
        datas = make_datas(event)

        # Create the eventcube
        eventcube = make_eventcube(data_root, event, datas)

        # check the eventcube for excess int16 values 
        print('>>>>>>>>>>>checking exceedence for= ',event.name )
        check_int16_exceedance(eventcube)

        # save datacube to data_root
        print(f"---Saving event datacube to: {event / f'datacube_{event.name}_{VERSION}.nc'}")
        eventcube.to_netcdf(event / f"datacube_{event.name}{VERSION}.nc")
        check_invalid_values(eventcube)
        print(f'>>>>>>>>>>>  eventcube finished for= {event.name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

        # TILE THE EVENT DATACUBE
        save_path = Path(event)
        # tile_datacube(eventcube, event, tile_size_x=256, tile_size_y=256 stride=256)

    print('>>>finished all events\n')
    print('>>>event_names= ',event_names)

if __name__ == "__main__":
    main()

