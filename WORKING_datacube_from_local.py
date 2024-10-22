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

def check_nan_gdal(tiff_path):
    '''
    Use GDAL to load and check TIFF for NaNs.
    '''
    dataset = gdal.Open(tiff_path)
    print(f"+++ in check_nan_gdal  +++")
    
    for band in range(1, dataset.RasterCount + 1):
        band_data = dataset.GetRasterBand(band).ReadAsArray()
        
        nan_count = np.isnan(band_data).sum()
        print(f"Band {band} NaN count: {nan_count}")
        
        nodata_value = dataset.GetRasterBand(band).GetNoDataValue()
        masked_values = np.count_nonzero(band_data == nodata_value)
        print(f"Band {band} Masked (nodata) count: {masked_values}")
    
    return dataset

def fill_nodata_with_zero(input_file):
    # Open the input raster file
    dataset = gdal.Open(input_file, gdal.GA_Update)
    band = dataset.GetRasterBand(1)

    # Get the nodata value (if it exists)
    nodata_value = band.GetNoDataValue()

    # Read the raster as a NumPy array
    data = band.ReadAsArray()

    # Replace nodata values with 0
    if nodata_value is not None:
        data[data == nodata_value] = 0
    
    # Write the modified array back to the raster
    band.WriteArray(data)
    
    # Flush and close the file
    band.FlushCache()
    dataset = None  # Close the file

# TODO add rtc function


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

def filter_allones(tile):
    # Example: After tiling, you can iterate through tiles to verify them
    for i, tile in enumerate(tiled_datacube):
        if np.all(tile == 1):
            print(f"Warning: Tile {i} has all values as 1. Check mask or data integrity.")

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

def tile_datacube(eventcube, tile_size_x=1024, tile_size_y=1024):
    '''
    Tile the datacube into smaller chunks (tiles) for efficient processing.
    The tiles will have size tile_size_x x tile_size_y.
    '''
    print(f"+++ Tiling Datacube +++")
    if not eventcube.rio.crs:
        raise ValueError("The datacube is missing a CRS. Please assign one before tiling.")

    
    # Rechunk the datacube with the specified tile sizes
    tiled_datacube = eventcube.chunk({'x': tile_size_x, 'y': tile_size_y, 'layer': 1})

    # Example: Write each tile to disk after chunking if necessary
    for tile in tiled_datacube:
        tile_path = f"output/tile_{i}.tif"
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

        missing_values = np.isnan(layer.values).sum()
        print(f"--- Missing Values (NaN): {missing_values}")

        # Check for Inf values
        inf_values = np.isinf(layer.values).sum()
        print(f"---Infinite Values (Inf): {inf_values}")

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
                print(f'---found file to reproject = {file.name}')
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

def match_resolutions_with_check(event):
    """
    Match the resolution and dimensions of the target raster to the reference raster
    only if they differ.
    """
    print('+++ in match_resolutions_with_check fn')

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
        if 'vv.tif' not in file.name and 'vh.tif' not in file.name:
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
    
    print("\n+++ Datacube Health Check Completed +++")



def main():

    VERSION = "Webb1"
    TRACKING = "id_Webb1"

    all_eventcubes = []  # List to hold datacubes for each event
    event_names = []  # List to hold event names
    # TODO add RTC functionality
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    for event in data_root.iterdir():
        # prepare the images
        # check if file is .nc
        if event.suffix == '.nc':
            continue
        if event.is_dir():
            print(f"---Preprocessing event: {event.name}")

            # Check all files for NaNs and fill with 0 if necessary
            for file in event.iterdir():
                if file.suffix == '.tif':
                    print('---checking= ', file.name)
                    filestr = str(file)
                    output_file = str(file.with_name(f'{file.stem}_nonans.tif'))
                    check_nan_gdal(filestr)
                    if file.name.endswith('slope.tif'):
                        # if nans exist, replace with 0
                        print('-------filling nans in slope.tiff----------------')
                        fill_nodata_with_zero(filestr)
                        print('---cheking nans in filled slope.tiff')
                        check_nan_gdal(filestr)
                    # split image into seperate vv and vh tiffs
                    create_vv_and_vh_tifs(file)

            # reproject everything to match higher resolution - needs to loop through whole event folder to find the vv.tif as reference
            match_resolutions_with_check(event)     
        # Get the datas info from the folder
        datas = make_datas(event)
        # Create the eventcube
        eventcube = make_eventcube(data_root, event, datas)
        print('--->>>>>>>>eventcube made for= ',event.name, '\n')
        # check the eventcube for excess int16 values 
        check_int16_exceedance(eventcube)
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
    # save datacube to data_root
    datacube.to_netcdf(data_root / f"---datacube_v{VERSION}.nc")
    check_int16_exceedance(datacube)

if __name__ == "__main__":
    main()

    # TODO add checks for failed eventdatacubes