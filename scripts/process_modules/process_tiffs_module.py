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
import json
# MODULES
# from check_int16_exceedance import check_int16_exceedance
import subprocess
import rasterio
import numpy as np
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.process_modules.process_helpers import  nan_check, print_dataarray_info



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

# SEPERATE SAR LAYERS
def create_vv_and_vh_tifs(file):
    '''
    will delete the original image after creating the vv and vh tifs
    '''
    print('+++in create_vv_and_vh_tifs fn')

    # print(f'---looking at= {file.name}')
    if 'img.tif' in file.name:
        #print(f'---found image file= {file.name}')
        # Open the multi-band TIFF
        with rasterio.open(file) as target:
            # Read the vv (first band) and vh (second band)
            vv_band = target.read(1)  # Band 1 (vv)
            vh_band = target.read(2)  # Band 2 (vh)
            # Define metadata for saving new files
            meta = target.meta
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




# CREAT ANALYSIS EXTENT
def create_extent_from_mask(mask_path, output_raster, no_data_value=None):
    # Load the mask file
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs

        # Identify no-data value
        if no_data_value is None:
            no_data_value = src.nodata
        if no_data_value is None:
            raise ValueError("No no-data value found in metadata or provided.")
            # create a binary mask with the entire image as 1
            

        # Create a binary mask (1 for valid data, 0 for no-data)
        binary_mask = (mask != no_data_value).astype(np.uint8)

    # Save the binary mask as a GeoTIFF
    with rasterio.open(
        output_raster,
        "w",
        driver="GTiff",
        height=binary_mask.shape[0],
        width=binary_mask.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(binary_mask, 1)

    print(f"extent saved to {output_raster}")


# NORMALIZING
def compute_image_min_max(image, band_to_read=1):
    with rasterio.open(image) as src:
        # Read the data as a NumPy array
        data = src.read(band_to_read)  # Read the first band
        # Update global min and max
        min = int(data.min())
        max = int(data.max())
        print(f"---{image.name}: Min: {data.min()}, Max: {data.max()}")
    return min, max


def calculate_and_normalize_slope(input_dem, mask_code):
    """
    Calculate slope from a DEM using GDAL and normalize it between 0 and 1.
    """

    # Step 1: Calculate slope using GDAL's gdaldem
    temp_slope = "temp_slope.tif"  # Temporary slope file
    gdal_command = [
        "gdaldem", "slope",
        input_dem,         # Input DEM
        temp_slope,         # Output raw slope file
        "-compute_edges"
    ]

    try:
        subprocess.run(gdal_command, check=True)
        print(f"Raw slope raster created: {temp_slope}")
    except subprocess.CalledProcessError as e:
        print(f"Error calculating slope: {e}")
        return

    # Step 2: Normalize slope using rasterio
    with rasterio.open(temp_slope) as src:
        slope = src.read(1)  # Read the slope data
        slope_min, slope_max = slope.min(), slope.max()
        print(f"Min slope: {slope_min}, Max slope: {slope_max}")

        # Normalize the slope to the range [0, 1]
        slope_norm_data = (slope - slope_min) / (slope_max - slope_min)

        # Prepare metadata for output file
        meta = src.meta.copy()
        meta.update(dtype='float32')

        normalized_slope = input_dem.parent / f"{mask_code}_slope_norm.tif"

        # Save the normalized slope
        with rasterio.open(normalized_slope, 'w', **meta) as dst:
            dst.write(slope_norm_data.astype(np.float32), 1)

    # Cleanup temporary raw slope file
    Path(temp_slope).unlink()

    print(f"Normalized slope raster saved to: {normalized_slope}")
    return normalized_slope



# CHANGE DATA TYPE
def make_float32_inf(input_tif, output_file):
    '''
    converts the tif to float32
    '''
    # print('+++in make_float32 inf')
    with rasterio.open(input_tif) as src:
        data = src.read()
        # print(f"---Original shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == 'float32':
            print(f'---{input_tif.name} already float32')
            meta = src.meta.copy()
            meta['count'] = 1
        else:
            # print(f'---{input_tif.name} converting to float32')
            # Update the metadata
            meta = src.meta.copy()
            meta.update(dtype='float32')
            # set num bands to 1
            meta['count'] = 1
            # Write the new file
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(data.astype('float32'))
            return output_file


def make_float32(input_tif, file_name):
    '''
    converts the tif to float32
    '''
    input_tif = Path(input_tif)
    file_name = Path(file_name)
    print('+++in make_float32 fn')
    with rasterio.open(input_tif) as src:
        data = src.read()
        print(f"---Original shape: {data.shape}, dtype: {data.dtype}")
        if data.dtype == 'float32':
            print(f'---{input_tif.name} already float32')
            src.close()
            input_tif.rename(file_name)
            print(f'---renamed {input_tif.name} to {file_name}')
            return file_name

        else:
            print(f'---{input_tif.name} converting to float32')
            # Update the metadata
            meta = src.meta.copy()
            meta.update(dtype='float32')
            # set num bands to 1
            meta['count'] = 1
            # Write the new file
            with rasterio.open(file_name, 'w', **meta) as dst:
                dst.write(data.astype('float32'))        
        return file_name


def make_float32_inmem(input_tif):

    # Open the input TIFF file
    with rasterio.open(input_tif) as src:
        # Read the data from the input file
        data = src.read()
        meta = src.meta.copy()

        # Check if data is already float32
        if meta['dtype'] == 'float32':
            print('---Data already in float32 format.')
            return src  # Return the original dataset if already float32

        # Convert data to float32
        converted_data = data.astype('float32')

        # Update metadata to reflect new dtype
        meta.update(dtype='float32')

        # Create a new in-memory file with updated metadata and float32 data
        with MemoryFile() as memfile:
            with memfile.open(**meta) as mem:
                mem.write(converted_data)
                print('---Converted to float32 and written to memory.')
                return memfile.open()


    return output_file

def xxx():

        # MATCH THE DEM TO THE SAR IMAGE
        # final_dem = extract_folder / f'{mask_code}_aligned_dem.tif'
        # match_dem_to_mask(image, dem, final_dem)
        # # print(f'>>>final_dem={final_dem.name}')

        # # CHECK THE NEW DEM
        # with rasterio.open(final_dem) as dem_src:
        #     with rasterio.open(image) as img_src:
        #         image_bounds = img_src.bounds
        #         image_crs = img_src.crs
        #         print(f'>>>image crs={image_crs}')
        #         print(f'>>>dem crs={dem_src.crs}')  
        #         img_width = img_src.width
        #         img_height = img_src.height
        #         img_transform = img_src.transform
        #         print(f'>>>bounds match={dem_src.bounds == image_bounds}')
        #         print(f'>>>crs match={dem_src.crs == image_crs}')
        #         print(f'>>>width match={dem_src.width == img_width}')
        #         print(f'>>>height match={dem_src.height == img_height}')
        #         print(f'>>>transform match={dem_src.transform == img_transform}')
        #         print(f'>>>count match={dem_src.count == img_src.count}')

        # normalized_slope = calculate_and_normalize_slope(final_dem, mask_code)

        # # CHECK THE SLOPE
        # with rasterio.open(extract_folder / f'{mask_code}_slope_norm.tif') as src:
        #     print(f'>>>slope min={src.read().min()}')
        #     print(f'>>>slope max={src.read().max()}')
        #     data = src.read()
        #     nonans = nan_check(data)
        #     print(f'>>>nonans?={nonans}')
        pass

# REPROJECTING
def reproject_layers_to_4326_TSX( src_path, dst_path):
    print('+++in reproject_layers_to_4326_TSX fn')

    with rasterio.open(src_path) as src:
        # print(f'---src_path= {src_path.name}')
        # print(f'---dst_path= {dst_path.name}')
        # print(f'---src_path crs = {src.crs}')
        
        transform, width, height = calculate_default_transform(src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': 'EPSG:4326', 'transform': transform, 'width': width, 'height': height, 'dtype': 'float32'})
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                band=src.read(i)
                band[band > 1] = 1
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)
            # print(f'---reprojected {src_path.name} to {dst_path.name} with {dst.crs}')

def reproject_to_4326_gdal(input_path, output_path, resampleAlg):
    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)
    # Open the input raster
    src_ds = gdal.Open(input_path)
    if not src_ds:
        print(f"Failed to open {input_path}")
        return
    
    # Ensure the input raster has a spatial reference system
    src_srs = src_ds.GetProjection()
    if not src_srs:
        raise ValueError(f"Input raster {input_path} has no CRS.")

    target_srs = 'EPSG:4326'

    # Use GDAL's warp function to reproject
    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,  # Target CRS
        resampleAlg=resampleAlg,  # Resampling method (nearest neighbor for categorical data)
        format="GTiff",
    )  
    gdal.Warp(output_path, src_ds, options=warp_options)
    # print(f"---Reprojected raster saved to: {output_path}")           

    return output_path   

def reproject_to_4326_fixpx_gdal(input_path, output_path, resampleAlg, px_size):
    print('+++in reproject_to_4326_fixpx_gdal fn')
    # print(f'---resampleAlg= {resampleAlg}')
    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)
    # Open the input raster
    src_ds = gdal.Open(input_path)
    if not src_ds:
        print(f"Failed to open {input_path}")
        return

    target_srs = 'EPSG:4326'

    # Use GDAL's warp function to reproject
    warp_options = gdal.WarpOptions(
        dstSRS=target_srs,  # Target CRS
        xRes=px_size,
        yRes=px_size,
        resampleAlg=resampleAlg,  # Resampling method (nearest neighbor for categorical data)
    )  
    gdal.Warp(output_path, src_ds, options=warp_options)
    # print(f"---Reprojected raster saved to: {output_path}")           

    return output_path   

# RESAMPLING
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

def resample_tiff(src_image, dst_image, target_res):
    print(f'+++resample_tiff::::target res= {target_res}')
    with rasterio.open(src_image, 'r') as src:
        # Read metadata and calculate new dimensions
        src_res = src.res  # (pixel width, pixel height in CRS units)
        print(f'---src_res= {src_res}')
        scale_factor_x = src_res[0] / target_res
        scale_factor_y = src_res[1] / target_res

        new_width = round(src.width * scale_factor_x)
        new_height = round(src.height * scale_factor_y)
        print(f'--- New Dimensions: width={new_width}, height={new_height}')
        new_transform = src.transform * src.transform.scale(scale_factor_x, scale_factor_x)
        print(f'--- New Transform: {new_transform}')

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
        })

        # Resample and write to the new file
        with rasterio.open(dst_image, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):  # Loop through each band
                resampled_data = src.read(
                    i,
                    out_shape=(new_height, new_width),
                    resampling=Resampling.bilinear
                )
                dst.write(resampled_data, i)
    print(f"Resampled image saved to {dst_image}")

def resample_tiff_gdal(src_image, dst_image, target_res):
    """
    Simplified resampling of a GeoTIFF to the specified resolution using GDAL.

    Parameters:
        src_image (str): Path to the source GeoTIFF.
        dst_image (str): Path to save the resampled GeoTIFF.
        target_res (float): Target resolution in CRS units (e.g., meters per pixel).
    """
        # Ensure inputs are strings
    if isinstance(src_image, Path):
        src_image = str(src_image)
    if isinstance(dst_image, Path):
        dst_image = str(dst_image)
    print(f'+++ Resampling {src_image} to target resolution: {target_res} m/pixel')

    # Open the source dataset
    src_ds = gdal.Open(src_image)
    if not src_ds:
        raise FileNotFoundError(f"Cannot open source file: {src_image}")

    # Use GDAL's Warp function to resample
    gdal.Warp(
        dst_image,
        src_ds,
        xRes=target_res,
        yRes=target_res,
        resampleAlg=gdal.GRA_Bilinear,  # Use bilinear resampling
        outputType=gdal.GDT_Float32,   # Save as 32-bit float
    )

    print(f'+++ Resampled image saved to: {dst_image}')

# CREATING DATACUBES

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
            # LOOP TIFFS + PREPARE THEM FOR DATA CUBE
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

def make_layerdict_TSX(extracted):
    '''
    iterates through the files in the event folder and returns a dict of the datas
    '''
    datas = {}
    for file in extracted.iterdir():
        # print(f'---file {file}')
        if 'final_image.tif' in file.name:
            datas[file.name] = 'hh'
            # print(f'---+image file found {file}')
        elif '4326_dem.tif' in file.name:
            datas[file.name] = 'dem'
            # print(f'---+dem file found {file}')
        elif '4326_slope.tif' in file.name:
            datas[file.name] = 'slope'   
            # print(f'---+slope file found {file}')
        elif 'final_mask.tif' in file.name:
            datas[file.name] = 'mask'
            # print(f'---+mask file found {file}')
        elif 'final_extent.tif' in file.name:
            datas[file.name] = 'extent'
            # print(f'---+valid file found {file}')

    #print('---datas ',datas)
    return datas

    print("\n+++ Datacube Health Check Completed +++")


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


def make_das_from_layerdict( layerdict, folder):
    dataarrays = []
    layer_names = []
    for tif_file, band_name in layerdict.items():
        # print(f'---tif_file= {tif_file}')
        # print(f'---band_name= {band_name}')
        filepath = folder / tif_file
        # print(f'---**************filepath = {filepath.name}')
        tiffda = rxr.open_rasterio(filepath)
        nan_check(tiffda)
        # print(f'---{band_name}= {tiffda}')   
        # check num uniqq values
        # print(f"---Unique data: {np.unique(tiffda.data)}")
        # print("----unique values:", np.unique(tiffda.values))
        dataarrays.append(tiffda)
        layer_names.append(band_name)
# 
    return dataarrays, layer_names
# def set_tif_dtype_to_float32(tif_file):
    
def check_int16_range(dataarray):
    # TAKES A DATAARRAY NOT A DATASET
    #print("+++in small int16 range check fn+++")
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if (dataarray < int16_min).any() or (dataarray > int16_max).any():
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        # Calculate actual min and max values in the array
        actual_min = dataarray.min().item()
        actual_max = dataarray.max().item()
        
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        print(f"---Minimum value found: {actual_min}")
        print(f"---Maximum value found: {actual_max}")
        return False
    
    # else:
    #     print(f"---no exceedances int16.")

    # Optional: Replace NaN and Inf values if necessary
    # dataarray = dataarray.fillna(0)  # Replace NaN with 0 or another appropriate value
    # dataarray = dataarray.where(~np.isinf(dataarray), 0)  # Replace Inf with 0 or appropriate value

def nan_check(nparray):
    if np.isnan(nparray).any():
        print("----Warning: NaN values found in the data.")
        return False
    else:
        print("----NO NANS FOUND")
        return True


def create_event_datacube_TSX(event, mask_code, VERSION="v1"):
    '''
    An xarray dataset is created for the event folder and saved as a .nc file.
    '''
    print(f'+++++++++++ IN CREAT EVENT DATACUBE TSX {event.name}+++++++++++++++++')
    # FIND THE EXTRACTED FOLDER
    extracted_folder = list(event.rglob(f'*{mask_code}_extracted'))[0]

    layerdict = make_layerdict_TSX(extracted_folder)

    print(f'---making das from layerdict= {layerdict}')
    dataarrays, layer_names = make_das_from_layerdict( layerdict, extracted_folder)

    # print(f'---CHECKING DATAARRAY LIST')
    # check_dataarray_list(dataarrays, layer_names)

    print(f'---CREATING CONCATERNATED DATASET')
    da = xr.concat(dataarrays, dim='layer').astype('float32')   
    da = da.assign_coords(layer=layer_names)

    # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
    if 'band' in da.dims and da.sizes['band'] == 1:
        print('---Squeezing out the "band" dimension')
        da = da.squeeze('band') 

    # print_dataarray_info(da)

    #######   CHUNKING ############
    # da = da.chunk({'x': 256, 'y': 256, 'layer': 1})
    # print('---Rechunked datacube')  

    #######   SAVING ############
    output_path = extracted_folder / f"{mask_code}.nc"
    da.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')
    
    print(f'>>>>>>>>>>>  ds saved for= {event.name} bye bye >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')


def create_event_datacube_TSX_inf(event, mask_code, VERSION="v1"):
    '''
    An xarray dataset is created for the event folder and saved as a .nc file.
    '''
    print(f'+++++++++++ IN CREAT EVENT DATACUBE TSX INF {event.name}+++++++++++++++++')
    # FIND THE EXTRACTED FOLDER
    print(f'---mask code= {mask_code}')
    extracted_folder = list(event.rglob(f'*{mask_code}_extracted'))[0]
    print(f'---extrated folder = {extracted_folder}')
    layerdict = make_layerdict_TSX(extracted_folder)

    print(f'---making das from layerdict= {layerdict}')
    dataarrays, layer_names = make_das_from_layerdict( layerdict, extracted_folder)

    print(f'---CHECKING DATAARRAY LIST')
    # check_dataarray_list(dataarrays, layer_names)
    # print('---dataarrays list = ',dataarrays) 
    print(f'---CREATING CONCATERNATED DATASET')
    da = xr.concat(dataarrays, dim='layer').astype('float32')   
    da = da.assign_coords(layer=layer_names)

    # If the 'band' dimension is unnecessary (e.g., single-band layers), squeeze it out
    if 'band' in da.dims and da.sizes['band'] == 1:
        # print('---Squeezing out the "band" dimension')
        da = da.squeeze('band') 

    # print_dataarray_info(da)

    #######   CHUNKING ############
    # da = da.chunk({'x': 256, 'y': 256, 'layer': 1})
    # print('---Rechunked datacube')  

    #######   SAVING ############
    output_path = extracted_folder / f"{mask_code}.nc"
    da.to_netcdf(output_path, mode='w', format='NETCDF4', engine='netcdf4')
    
    print(f'##################  ds saved for= {event.name} bye bye #################\n')


# WORK ON DEM
def match_dem_to_mask(sar_image, dem, output_path):
    """
    Matches the DEM to the SAR image grid by enforcing exact alignment of transform, CRS, and dimensions.
    """
    print('+++in match_dem_to_sar fn')
    

    output_path.unlink(missing_ok=True)  # Deletes the file if it exists
    # Open the SAR image to extract its grid and CRS
    with rasterio.open(sar_image) as sar:
        sar_transform = sar.transform
        sar_crs = sar.crs
        print(f"---SAR CRS: {sar_crs}")
        sar_width = sar.width
        sar_height = sar.height

    # Open the DEM to reproject and align it
    with rasterio.open(dem) as dem_src:
        print(f"---DEM CRS: {dem_src.crs}")
        dem_meta = dem_src.meta.copy()
        # Update DEM metadata to match SAR grid
        dem_meta.update({
            'crs': sar_crs,
            'transform': sar_transform,
            'width': sar_width,
            'height': sar_height,
            'dtype': "float32"
        })
        with rasterio.open(output_path, 'w', **dem_meta) as dst:
            print(f'---output_path= {output_path.name}')
            # Reproject each band of the DEM
            for i in range(1, dem_src.count + 1):
                reproject(
                    source=rasterio.band(dem_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=sar_transform,
                    dst_crs=sar_crs,
                    resampling=Resampling.nearest  # Nearest neighbor for discrete data like DEM
                )

    print(f"Reprojected and aligned DEM saved to: {output_path}")


def create_slope_from_dem(target_file, dst_file):
    cmd = [
    "gdaldem", "slope", f"{target_file}", f"{dst_file}", "-compute_edges",
    "-p"
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("Slope calculation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# CLIPPING AND ALIGNMENT
def clip_image_to_mask_gdal(input_raster, mask_raster, output_raster):

    # Open the mask to extract its bounding box
    mask_ds = gdal.Open(mask_raster)
    if mask_ds is None:
        raise FileNotFoundError(f"Mask file not found: {mask_raster}")
    mask_transform = mask_ds.GetGeoTransform()
    mask_proj = mask_ds.GetProjection()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize

    print(f"---Mask dimensions: width={mask_width}, height={mask_height}")

    # Configure warp options to match the mask's resolution and extent
    options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(mask_transform[0], 
                      mask_transform[3] + mask_transform[5] * mask_height, 
                      mask_transform[0] + mask_transform[1] * mask_width, 
                      mask_transform[3]),
        xRes=mask_transform[1],  # Pixel size in X
        yRes=abs(mask_transform[5]),  # Pixel size in Y
        dstSRS=mask_proj,  # Match CRS
        resampleAlg="nearest",  # Nearest neighbor for categorical data
        outputBoundsSRS=mask_proj  # Ensure alignment in mask CRS
    )
    gdal.Warp(output_raster, input_raster, options=options)

    print(f"Clipped raster saved to: {output_raster.name}")
    mask_ds = None  # Close the mask dataset
    # Delete the original SAR image
    Path(input_raster).unlink()


    print(f"Clipped raster saved to: {output_raster}")

def clean_mask(mask_path, output_path):
        with rasterio.open(mask_path) as src:
            data = src.read(1)
            print(f">>> Original mask stats: min={data.min()}, max={data.max()}, unique={np.unique(data)}")

            meta = src.meta.copy()
            # remove numbers greater than 1
            data[data > 1] = 0
            print(f"--- Modified mask unique values: {np.unique(data)}")

            assert (data.min() == 0 and data.max() == 1) and len(np.unique(data)) == 2
            meta.update(dtype='uint8')  # Ensure uint8 format

            # Write the cleaned mask
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data.astype('uint8'), 1)  # Ensure uint8 format

        print(f"Cleaned and aligned mask saved to: {output_path}")

def align_image_to_mask(sar_image, mask, aligned_image):
    print('+++in align_image_to_mask fn')

    # Open the mask to get CRS, transform, and dimensions
    with rasterio.open(mask) as mask_src:
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform
        mask_width = mask_src.width
        mask_height = mask_src.height
        mask_res = (abs(mask_transform[0]), abs(mask_transform[4]))  # Ensure positive resolution
        print('---mask ok')

    # Open the SAR image to calculate alignment
    with rasterio.open(sar_image) as sar_src:
        sar_meta = sar_src.meta.copy()
        transform, width, height = calculate_default_transform(
            sar_src.crs, mask_crs, sar_src.width, sar_src.height, *sar_src.bounds, resolution=mask_res
        )
        sar_meta.update({
            'crs': mask_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        print('---sar ok')

        # Write the aligned SAR image
        with rasterio.open(aligned_image, 'w', **sar_meta) as aligned_dst:
            for i in range(1, sar_src.count + 1):
                reproject(
                    source=rasterio.band(sar_src, i),
                    destination=rasterio.band(aligned_dst, i),
                    src_transform=sar_src.transform,
                    src_crs=sar_src.crs,
                    dst_transform=transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.nearest
                )

    print(f"---Aligned SAR image saved to: {aligned_image}")


# probably not needed
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
            

