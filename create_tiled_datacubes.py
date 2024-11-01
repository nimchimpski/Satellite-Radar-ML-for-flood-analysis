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
from modules.helpers import *
from tile_datacube import tile_datacube 
'''
- will overwrite existing .nc files in the event folders.
- expects each event folder to have the following files:
    elevation.tif
    slope.tif
    msk.tif
    valid.tif

TODO check process in place to organise the dataroot / input files such as they are correct for this script.
TODO THERE IS A PERSISTANT HIDDEN PROBLEM WITH WRONG VALUES AFTER CASTING.
TODO add rtc function 
'''

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

def make_ds(data_root, event, datas):
    '''
    MAKES AN XARRAY 'DATASET' (!!!)
    THIS MEANS 1ST DIMENSION IS A VARIABLE NAME. DEFAULT = '--xarray_dataarray_variable--'
    THIS MEANS FURTHER VARIABLES CAN BE ADDED - EG TIME SERIES DATA FOR SAME AREA
    'Event' is a single folder conatining flood event data    
    'satck' is an Xarray DataArray 
    '''   
    print(f"++++ in single ds fn")      
    layers = []
    layer_names = []
    # CREATE THE DATACUBE
    # TODO SKIP XML's etc to avoid error
    for tif_file, band_name in tqdm(datas.items(), desc='---making da cubes from bands in "datas"'):
        try:
            #print(f"\n---Loading {tif_file}-------------------------------------------")
            # 'da'  IS A DATA ARRAY

            # CREATE A DATA ARRAY FROM THE TIF FILE
            da = rxr.open_rasterio(Path(data_root, event, tif_file))
            # Check if da is created successfully
            if da is None:
                print(f"---Error: da for {tif_file} is None")

        except Exception as e:
            print(f"eee-Error loading {tif_file}: {e}")
            continue  # Skip this file if there is an issue loading
        # Handle multi-band image data (e.g., VV, VH bands)
    
        try:
            #print(f"---Processing single-band layer: {band_name}")
            layers.append(da)
            layer_names.append(band_name)
            #print(f"---Successfully processed layer with band name: {band_name}")
        except Exception as e:
            print(f"eee-Error creating layers for {tif_file}: {e}")
            continue
    if len(layers) != 6:
        print('---length layers= {len(layers)} not 6')         
    #print(f'---layer_names= {layer_names}')
    print(f'---finished {event.name}\n'  )   

    if layers:
        # REPROJECT ALL LAYERS TO MATCH THE FIRST LAYER
        reprojected_layers = []
        for layer in tqdm(layers, desc='---reprojecting'):
            reprojected_layer = layer.rio.reproject_match(layers[0])
            reprojected_layers.append(reprojected_layer)

        # CONCATENATE THE LAYERS ALONG A NEW DIMENSION CALLED 'layer'
        ds = xr.concat(layers, dim='layer').astype('int16')

        # Assign the layer names (e.g., ['vv', 'vh', 'dem', 'slope']) to the 'layer' dimension
        ds = ds.assign_coords(layer=layer_names)
        print('---layer names assigned ok')

        # MAKE da INTO A DATASET !!!!!!!!!!!!
        ds = ds.to_dataset(name="data1")  # Replace with a suitable name

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

        return ds
    else:
        print('---No layers found')
        return None 
    
def make_datas(event):
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

def create_event_datacubes(data_root, VERSION="v1"):
    '''
    data_root: Path to the root directory containing the event folders.
    An xarray dataset is created for each event folder and saved as a .nc file.
    NB DATASET IS A COLLECTION OF DATAARRAYS
        # TODO add RTC functionality

    '''
    # LOOP THE FOLDERS BY  COUNTRIES, SOME REPEATED WITH DIF NUMBERED SUFFIX 
    for event in data_root.iterdir(): # CREATES ITERABLE FO FILE PATHS 

        # IGNORE .NC FILES
        if event.suffix == '.nc' or event.name in ['tiles'] or not event.is_dir():
            # print(f"Skipping file: {event.name}")
            continue

        if event.is_dir() and any(event.iterdir()):
            print(f"***************** {event.name}   PREPARING TIFS *****************: ")
            # DO WORK ON THE TIFS

            # TODO FIX THIS MESS:

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

        print('##################### MAKING SINGLE CUBE ##############################')
        # Create the ds
        ds = make_ds(data_root, event, datas)
        print('################### CRS CHECK ##############################')
        crs = ds.rio.crs
        print('---ds crs = ',crs)

        # check the ds for excess int16 values 
        #print('>>>>>>>>>>>checking exceedence for= ',event.name )
        
            # Iterate over each variable in the dataset
        for var_name, dataarray in ds.data_vars.items():
            print(f"---Checking variable: {var_name}")
            check_int16_range(dataarray)

        # Ensure CRS is applied to the dataset using `rio.write_crs`
        ds.rio.write_crs("EPSG:4326", inplace=True)   
        print('---ds crs = ',ds.rio.crs)


        # assign the  crs to the ds !!!!
        ds['spatial_ref'] = xr.DataArray(0, attrs={
            'grid_mapping_name': 'latitude_longitude',
            'epsg_code': crs.to_epsg(),
            'crs_wkt': ds.rio.crs.to_wkt()
            })  
        
        # # Link the CRS to the main data variable
        # ds['data1'].attrs['grid_mapping'] = 'spatial_ref'

        print('################### SAVING DS ##############################')

        # save datacube to data_root
        print(f"---Saving event datacube : {f'datacube_{event.name}_{VERSION}.nc'}")
        output_path = event / f"datacube_{event.name}{VERSION}.nc"
        ds.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')

        # Check CRS after reopening
        with xr.open_dataset(output_path) as datacube:
            print("---CRS after reopening:", datacube.rio.crs)
            # TODO STILL NO CRS IN THE REOPENED DATACUBE
        
        # TODO COLLAPSE THESE FUNCTIONS
        for var_name, dataarray in ds.data_vars.items():
            print(f"Checking variable: {var_name}")
            nan_check(dataarray)
        for var_name, dataarray in ds.data_vars.items():
            print(f"Checking variable: {var_name}")
            check_int16_range(dataarray)

        print(f'>>>>>>>>>>>  ds saved for= {event.name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')


    print('>>>finished all events\n')

def main():
    data_root = Path(r"\\cerndata100\AI_files\Users\AI_flood_service\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    create_event_datacubes(data_root)

    for event in data_root.iterdir():
       if event.is_dir() and any(event.iterdir()): # select event folders
           print(f"############## {event.name}   TILING ########################: ")
           for file in tqdm(event.iterdir(), desc=':::::iterate files in event folders'):
               if file.suffix == '.nc':
                   #datacube = xr.open_dataset(file)
                   #print('---datacube= ',datacube)
                   print(f">>>>>>>>>>>>>>>>>>>>>>> TILING {event.name}<<<<<<<<<<<<<<<<: ")
                   total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask = tile_datacube(file, event, tile_size=256, stride=256)

    print(f'>>>>total num of tiles: {total_num_tiles}')
    print(f'>>>>num of saved tiles: {num_saved}')
    print(f'>>>num witno valid data/analysis extent layer: {num_has_nans}')
    print(f'>>>num with no mask : {num_nomask}')
    print(f'>>>num with no valid layer : {num_novalid}')

if __name__ == "__main__":
    main()

