from pathlib import Path
import shutil
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.preprocess_modules.process_tiffs_module import match_dem_to_sar
from scripts.preprocess_modules.process_tiffs_module import calculate_and_normalize_slope, create_event_datacubes_TSX,         reproject_layers_to_match_TSX, nan_check


data_root = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_process1\archive')


# if False:
if True:
    # MOVE THE MASK, IMAGE, AND DEM TO THE EXTRACTED FOLDER
    for event in data_root.iterdir():
        print(f'################### EVENT={event.name}  ###################')
        extract_folder = event / 'extracted'
        if extract_folder.exists():
            shutil.rmtree(extract_folder)
        extract_folder.mkdir(exist_ok=True)

        # COPY THE MASK
        mask = list(event.rglob('*MASK.tif'))[0]
        print(f'>>>mask={mask.name}')
        if not (extract_folder / mask.name).exists():
            shutil.copy(mask, extract_folder)

        # GET REGION CODE FROM MASK
        mask_code = mask.name.split('_')[0]
        print(f'>>>mask_code= ',mask_code)

        # COPY THE SAR IMAGE
        image = list(event.rglob('*IMAGE*.tif'))[0]
        print(f'>>>imagename ={image.name}')
        image_rename = extract_folder / f'{mask_code}_image.tif'
        print(f'>>>image_rename={image_rename.name}')
        if not image_rename.exists():
            shutil.copy(image, image_rename)

        # COPY THE DEM
        dem = list(event.rglob('*DEM*.tif'))[0]
        print(f'>>>dem={dem.name}')
        if not (extract_folder / dem.name).exists():
            shutil.copy(dem, extract_folder)

        # MATCH THE DEM TO THE SAR IMAGE
        final_dem = extract_folder / f'{mask_code}_dem.tif'
        match_dem_to_sar(image, dem, final_dem)
        # print(f'>>>final_dem={final_dem.name}')

        # CHECK THE NEW DEM
        with rasterio.open(final_dem) as dem_src:
            with rasterio.open(image) as img_src:
                image_bounds = img_src.bounds
                image_crs = img_src.crs
                print(f'>>>image crs={image_crs}')
                print(f'>>>dem crs={dem_src.crs}')  
                img_width = img_src.width
                img_height = img_src.height
                img_transform = img_src.transform
                print(f'>>>bounds match={dem_src.bounds == image_bounds}')
                print(f'>>>crs match={dem_src.crs == image_crs}')
                print(f'>>>width match={dem_src.width == img_width}')
                print(f'>>>height match={dem_src.height == img_height}')
                print(f'>>>transform match={dem_src.transform == img_transform}')
                print(f'>>>count match={dem_src.count == img_src.count}')

        normalized_slope = calculate_and_normalize_slope(final_dem, mask_code)

        # CHECK THE SLOPE
        with rasterio.open(extract_folder / f'{mask_code}_slope_norm.tif') as src:
            print(f'>>>slope min={src.read().min()}')
            print(f'>>>slope max={src.read().max()}')
            data = src.read()
            nonans = nan_check(data)
            print(f'>>>nonans?={nonans}')

        output_path = extract_folder / f'{mask_code}_mask_clean.tif'
        # CHECK THE MASK AND REMOVE THE 3'S
        with rasterio.open(mask) as src:
            # Read the mask data and metadata
            data = src.read(1)
            meta = src.meta.copy()
            print(f">>> Original mask stats: min={data.min()}, max={data.max()}, unique={np.unique(data)}")

            # Replace 3s with 0s
            data[data == 3] = 0
            print(f"--- Modified mask unique values: {np.unique(data)}")

            # Check and preserve transform, CRS, and metadata
            meta.update(dtype="uint8")  # Set data type to Byte

            # Write the cleaned mask
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data.astype("uint8"), 1)  # Ensure uint8 format

        print(f"Cleaned and aligned mask saved to: {output_path}")

            # # CHECK THE NEW MASK
            # with rasterio.open(final_mask) as msrc:
            #     unique_values = np.unique(data)
            #     print(f"Unique values in mask: {unique_values}")

        break


        # REPROJECT THE TIFS TO EPSG:4326
        print('\n>>>>>>>>>>>>>>>> reproj all layers to 4326 >>>>>>>>>>>>>>>>>')
        reprojected_image = extract_folder / f'{mask_code}_4326_image.tif'
        reprojected_dem = extract_folder / f'{mask_code}_4326_dem.tif'
        reprojected_slope = extract_folder / f'{mask_code}_4326_slope.tif'
        reprojected_mask = extract_folder / f'{mask_code}_4326_mask.tif'

        # REPROJECT THE IMAGE TO EPSG:4326
        # with rasterio.open(image) as img_src:
        #     transform, width, height = calculate_default_transform(img_src.crs, 'EPSG:4326', img_src.width, img_src.height, *img_src.bounds)
        #     kwargs = img_src.meta.copy()
        #     kwargs.update({'crs': 'EPSG:4326', 'transform': transform, 'width': width, 'height': height})
        #     with rasterio.open(reprojected_image, 'w', **kwargs) as dst:
        #         for i in range(1, img_src.count + 1):
        #             reproject(
        #                 source=rasterio.band(img_src, i),
        #                 destination=rasterio.band(dst, i),
        #                 src_transform=img_src.transform,
        #                 src_crs=img_src.crs,
        #                 dst_transform=transform,
        #                 dst_crs='EPSG:4326',
        #                 resampling=Resampling.nearest)
        
        # check iMAGE CRS
        # with rasterio.open(reprojected_image) as src:
        #     print(f'>>>image crs={src.crs}')


        orig_images = [ image, final_dem, normalized_slope, mask]
        rep_images = [reprojected_image, reprojected_dem, reprojected_slope, reprojected_mask]

        for i,j in zip( orig_images, rep_images):
            print(f'---i={i.name} j={j.name}')
            reproject_layers_to_match_TSX(i, j)

# print('\n>>>>>>>>>>>>>>>> create event datacubes >>>>>>>>>>>>>>>>>')
# create_event_datacubes_TSX(data_root)

