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
import shutil
import re
# MODULES
# from check_int16_exceedance import check_int16_exceedance
from ..modules.helpers import *
from ..modules.organise_folders import collect_images


def fill_nodata_with_zero(input_file):
    print('+++in fill_nodata_with_zero fn')
    #print('+++in fill_nodata_with_zero fn')
    input_str = str(input_file)
    # Open the input raster file
    dataset = gdal.Open(input_str, gdal.GA_Update)
    band = dataset.GetRasterBand(1)

    # Get the nodata value (if it exists)
    nodata_value = band.GetNoDataValue()

    # Read the raster as a NumPy array
    data = band.ReadAsArray()

    # Replace nodata values with 0
    if nodata_value is not None:
        #print('---replacing nans with 0 in ',input_file.name)
        data[data == nodata_value] = 0
    
    # Write the modified array back to the raster
    band.WriteArray(data)
    
    # Flush and close the file
    band.FlushCache()
    dataset = None  # Close the file

def check_layers(layers, layer_names):
    '''
    checks the layers and prints out some info
    '''
    print('\n+++in check_layers fn+++++++++++++++++++++++++')
    # Assuming you have a list of Dask arrays, each representing a layer
    
    for i, layer in enumerate(layers):
        print(f'---layer name = {layer_names[i]}')
        #print(f"---Layer {i+1}:")

        # Print the shape of the layer
        #print(f"---Shape: {layer.shape}")

        # Print the data type of the layer
        #print(f"---Data Type: {layer.dtype}")

        # Assuming the array has x and y coordinates in the `.coords` attribute (like in xarray)
        # You can access and print the coordinates if it is an xarray DataArray or a similar structure
        #if hasattr(layer, 'coords'):  # If using Dask with xarray-like data
        #    print(f"---X Coordinates: {layer.coords['x']}")
        #    print(f"---Y Coordinates: {layer.coords['y']}")
        #else:
        #    print("---No coordinate information available.")

        # Check for NaN or Inf values in the layer
        nan_check(layer)
        check_int16_range(layer)


def create_vv_and_vh_tifs(file):
    '''
    will delete the original image after creating the vv and vh tifs
    '''
    print('+++in create_vv_and_vh_tifs fn')

    # print(f'---looking at= {file.name}')
    if 'img.tif' in file.name:
        #print(f'---found image file= {file.name}')
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
            with rasterio.open(file.parent / vv_newname, 'w', **meta) as destination:
                destination.write(vv_band, 1)  # Write band 1 (vv)
            # Save the vh band as a separate TIFF
            vh_newname = file.name.rsplit('_', 1)[0]+'_vh.tif'
            with rasterio.open(file.parent / vh_newname, 'w', **meta) as destination:
                destination.write(vh_band, 1)  # Write band 2 (vh)
        print('---DELETING ORIGINAL IMAGE FILE')
        file.unlink()  # Delete the original image file  
    else:
        print(f'---NO IMAGE FOUND !!!!!= {file.name}')

            # delete original image using unlink() method
    #print('---finished create_vv_and_vh_tifs fn')


def match_resolutions_with_check(event):
    """
    Match the resolution and dimensions of the target raster to the reference raster
    only if they differ.
    """
    #print('+++++++in match_resolutions_with_check fn')

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
        #print(f'--- Reference file {reference_path.name} resolution: {reference_resolution}')
    
    # Loop through other files to check and reproject if necessary
    for file in event.iterdir():
        if file.is_file(): 
            patterns = ['vv.tif', 'vh.tif', '_s2_', '.json','.nc']
            if not any(i in file.name for i in patterns) and 'epsg4326' in file.name:

                #print(f'--- analysing this file = {file.name}')
                # Open the target layer to compare resolutions
                with rxr.open_rasterio(file) as target_layer:
                    target_resolution = target_layer.rio.resolution()
                    #print(f'--- Target file {file.name} resolution: {target_resolution}')

                    # Compare resolutions with a small tolerance
                    if abs(reference_resolution[0] - target_resolution[0]) < 1e-10 and \
                       abs(reference_resolution[1] - target_resolution[1]) < 1e-10:
                        #print(f"--- Skipping {file.name} (resolution already matches)")
                        continue

                    # Reproject target file to match reference
                    reprojected_layer = target_layer.rio.reproject_match(reference_layer)
                # Save reprojected file (once target file is closed)
                reprojected_layer.rio.to_raster(file)
                #print(f'--- Resampled raster saved to: {file.name}')

def make_dataset(data_root, event, datas):
    '''
    MAKES AN XARRAY 'DATASET' (!!!)
    THIS MEANS 1ST DIMENSION IS A VARIABLE NAME. DEFAULT = '--xarray_dataarray_variable--'
    THIS MEANS FURTHER VARIABLES CAN BE ADDED - EG TIME SERIES DATA FOR SAME AREA
    'Event' is a single folder conatining flood event data    
    'satck' is an Xarray DataArray 
    '''   
    print(f"\n++++ in make_dataset fn")      
    layers = []
    layer_names = []
    # CREATE THE DATACUBE
    # TODO SKIP XML's etc to avoid error
    for tif_file, band_name in tqdm(datas.items(), desc='---making lists of tiffs and corresponding band names'):
        try:
            #print(f"\n---Loading {tif_file}-------------------------------------------")
            # 'da'  IS A DATA ARRAY

            # CREATE A DATA ARRAY FROM THE TIF FILE
            tiffda = rxr.open_rasterio(Path(data_root, event, tif_file))
            # Check if da is created successfully
            if tiffda is None:
                print(f"---Error: da for {tif_file} is None")

        except Exception as e:
            print(f"---Error loading {tif_file}: {e}")
            continue  # Skip this file if there is an issue loading
        # Handle multi-band image data (e.g., VV, VH bands)
    
        try:
            #print(f"---Processing single-band layer: {band_name}")
            layers.append(tiffda)
            layer_names.append(band_name)
            #print(f"---Successfully processed layer with band name: {band_name}")
        except Exception as e:
            print(f"---Error creating layers for {tif_file}: {e}")
            continue
    if len(layers) != 6:
        print('---length layers= {len(layers)} not 6')         
    print(f'---layer_names created= {layer_names}')

    if layers:
        # REPROJECT ALL LAYERS TO MATCH THE FIRST LAYER
        reprojected_layers = []
        for layer in tqdm(layers, desc='---reprojecting'):
            reprojected_layer = layer.rio.reproject_match(layers[0])
            reprojected_layers.append(reprojected_layer)

        # CREATE NEW DATASET + CONCATENATE THE LAYERS ALONG A NEW DIMENSION CALLED 'layer'
        da = xr.concat(layers, dim='layer').astype('int16')

        print('---da is a dataset=',isinstance(da, xr.Dataset))
        print('---da layers =',da['layer'])

        # Assign the layer names (e.g., ['vv', 'vh', 'dem', 'slope']) to the 'layer' dimension
        da = da.assign_coords(layer=layer_names)
        print('---da layers after assigning=',da['layer'])


        print('---layer names finished assigning - did i fuck up?')

        # MAKE da INTO A DATASET !!!!!!!!!!!!
        ds = da.to_dataset(name="data1")  # Replace with a suitable name
        print('---ds made')
        print('---ds is a dataset???=',isinstance(ds, xr.Dataset))

        # Rechunk the final datacube for optimal performance
        ds = ds.chunk({'x': 1024, 'y': 1024, 'layer': 1})
        print('---Rechunked datacube')

        # Check if layers[0] has a CRS
        if layers[0].rio.crs is None:
            print("---Warning: layers[0] does not have a CRS. Assigning a default CRS.")
            # Optionally assign a default CRS, like EPSG:4326
            layers[0].rio.write_crs("EPSG:4326", inplace=True)
        else:
            print(f"---CRS for layers[0]: {layers[0].rio.crs}")

        ds.rio.write_crs(layers[0].rio.crs, inplace=True)


        # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
        if 'band' in ds.dims and ds.sizes['band'] == 1:
            ds = ds.squeeze('band')
            #print('---Squeezed out single-band dimension')

        print(f'---ds {event.name} made ok')
        print('---ds is a dataset=',isinstance(ds, xr.Dataset))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        return ds
    else:
        print('---No layers found')
        return None 
    
def make_datas(event):
    '''
    iterates through the files in the event folder and returns a dict of the datas
    '''
    datas = {}
    for file in event.iterdir():
        # print(f'---file {file}')
        if 'epsg4326' in file.name and '_s2_' not in file.name:
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
    #print('---datas ',datas)
    return datas



    
    print("\n+++ Datacube Health Check Completed +++")

def create_event_datacubes(data_root, save_path, VERSION="v1"):
    '''
    data_root: Path directory containing the event folders (1deep).

    An xarray dataset is created for each event folder and saved as a .nc file.
    NB DATASET IS A COLLECTION OF DATAARRAYS
        # TODO add RTC functionality

    '''
    # LOOP THE FOLDERS BY  COUNTRIES, SOME REPEATED WITH DIF NUMBERED SUFFIX 
    for event in tqdm(data_root.iterdir() ): # CREATES ITERABLE FO FILE PATHS 

        # IGNORE .NC FILES
        if event.suffix == '.nc' or event.name in ['tiles'] or not event.is_dir():
            # print(f"Skipping file: {event.name}")
            continue

        if event.is_dir() and any(event.iterdir()):
            print(f"***************** {event.name}   PREPARING TIFS *****************: ")
            # DO WORK ON THE TIFS
            # LOOP FILES + PREPARE THEM FOR DATA CUBE
            for file in event.iterdir():
                if 'epsg4326' in file.name and '_s2_' not in file.name:
                    if file.name.endswith('slope.tif') :
                            fill_nodata_with_zero(file)

            for file in event.iterdir():
                if '_s2_' not in file.name and 'epsg4326'  in file.name:
                    if file.name.endswith('img.tif'):
                        # EXTRACT THE VV AND VH BANDS FROM THE IMAGE FILE
                        # TODO this is getting called twice - check if file gets/needs deleting 
                        create_vv_and_vh_tifs(file)

        # Get the datas info from the folder
        datas = make_datas(event)

        print(f'##################### MAKING SINGLE DATASET {event.name} ##############################')

        # Create the ds
        ds = make_dataset(data_root, event, datas)
        print('---ds is a dataset=',isinstance(ds, xr.Dataset))

        # check the ds for excess int16 values 
        print('>>>>>>>>>>>checking exceedence for= ',event.name )
            # Iterate over each variable in the dataset
        for var_name, dataarray in ds.data_vars.items():
            print(f"---Checking variable: {var_name}")
            check_int16_range(dataarray)


        # Check and assign CRS to the dataset
        print(f'################### CRS CHECK {event.name} ##############################')
        # Step 1: Ensure CRS is applied to the dataset
        if not ds.rio.crs:
            ds.rio.write_crs("EPSG:4326", inplace=True)  # Replace with desired CRS
        print('---ds crs = ', ds.rio.crs)

        # Step 2: Add spatial_ref coordinate
        crs = ds.rio.crs
        ds['spatial_ref'] = xr.DataArray(
            0,
            attrs={
                'grid_mapping_name': 'latitude_longitude',
                'epsg_code': crs.to_epsg() if crs.to_epsg() else "Unknown EPSG",
                'crs_wkt': crs.to_wkt()
            }
        )
        print('---ds[spatial ref]',ds['spatial_ref'])
        print('---ds[spatial ref] attrs', ds['spatial_ref'].attrs)

        # Step 3: Link spatial_ref to 'data1'
        ds['data1'].encoding['grid_mapping'] = 'spatial_ref'



        print(f',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
        print('---dataset= ',ds)
        print(f',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
        
        # # Link the CRS to the main data variable
        # ds['data1'].attrs['grid_mapping'] = 'spatial_ref'

        print(f'################### SAVING {event.name} DS ##############################')

        # save datacube to data_root
        print(f"---Saving event datacube : {f'datacube_{event.name}_{VERSION}.nc'}")


        output_path = save_path / event.name / f"datacube_{event.name}{VERSION}.nc"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            print(f"---Overwriting existing file: {output_path}")
        else:
            print(f"---creating save folder: {output_path}")
        ds.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')

        # Check CRS after reopening
        print('---ds.attrs= ',ds['spatial_ref'].attrs['epsg_code'])
        print(ds['spatial_ref'].attrs.get("epsg_code", "Attribute not found"))
        print('---bye bye')

        
        # TODO COLLAPSE THESE FUNCTIONS
        for var_name, dataarray in ds.data_vars.items():
            print(f"Checking variable nan: {var_name}")
            nan_check(dataarray)
        for var_name, dataarray in ds.data_vars.items():
            print(f"Checking variable int16: {var_name}")
            check_int16_range(dataarray)

        print(f'>>>>>>>>>>>  ds saved for= {event.name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')


    print('>>>finished all events\n')

def process_terraSARx_data(data_root):
    '''
    makes a 'datacube_files' folder in each event folder and copies the necessary files to it
    '''
    print('+++in process_terraSARx_data fn')
    #image = list(Path('.').rglob("IMAGE_HH_*"))
    #print('---image= ',image)

    target_filename = "DEM_MAP.tif"

    for event in data_root.iterdir():
        if event.is_dir() and any(event.iterdir()):
            print(f"******* {event.name}   PREPARING TIFS ********")
            datacube_files_path = event / 'datacube_files'
            if datacube_files_path.exists() :
                shutil.rmtree(datacube_files_path)  # Delete the directory and all its contents

    
            datacube_files_path.mkdir(parents=True, exist_ok=True)
            pattern = re.compile(f'^{re.escape(target_filename)}$')
            # STEP THROUGH FILENAMES WE WANT 
            filename_parts = ['DEM_MAP', 'IMAGE_HH']
            for i in filename_parts:
                # print(f'---looking for files starting with {i}')
                pattern = re.compile(f'^{re.escape(i)}')  
                # COPY THEM TO THE EVENT DATA CUBE FOLDER
                for file_path in Path(event).rglob("*"):
                    if file_path.suffixes == ['.tif'] and pattern.match(file_path.name):
                    # if file_path.suffix != '.aux.xml'and file_path.name == target_filename:
                    # if True:    
                        target = datacube_files_path / file_path.name
                        # if Path(target).exists():
                        #     print('---file already exists') 
                        #     continue
                        shutil.copy(file_path, target)
            




