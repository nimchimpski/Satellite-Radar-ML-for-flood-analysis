from pathlib import Path
import shutil
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.preprocess_modules.process_tiffs_module import match_dem_to_sar
from scripts.preprocess_modules.process_tiffs_module import calculate_and_normalize_slope, create_event_datacubes_TSX,         reproject_layers_to_4326_TSX, nan_check, clip_image_to_mask, align_image_to_mask


data_root = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_process1\archive')

tiffs = 1

# MOVE THE MASK, IMAGE, AND DEM TO THE EXTRACTED FOLDER
for event in data_root.iterdir():
    if tiffs:    
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

        #############################################

        # GET REFERENCE DATA 
        with rasterio.open(image) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height

        # CLIP THE MASK TO THE SAR IMAGE
        final_mask = extract_folder / f'{mask_code}_mask_clean.tif'
        aligned_image = extract_folder / f'{mask_code}_aligned_image.tif'

        clip_image_to_mask(image, mask, aligned_image)

        # align_image_to_mask(image, mask, aligned_image)

        break

        # SORT THE MASK TO 1 AND 0
        # CHECK THE MASK AND REMOVE THE 3'S and align to the SAR image
        with rasterio.open(mask) as src:
            data = src.read(1).astype("float32")
            print(f">>> Original mask stats: min={data.min()}, max={data.max()}, unique={np.unique(data)}")
        #     transform, width, height = calculate_default_transform(
        #     src.crs, ref_crs, src.width, src.height, *src.bounds
        # )
            meta = src.meta.copy()
            meta.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height,
            'dtype': "float32"
            })

            # Replace 3s with 0s
            data[data == 3] = 0
            print(f"--- Modified mask unique values: {np.unique(data)}")

            # Write the cleaned mask
            with rasterio.open(final_mask, "w", **meta) as dst:
                reproject(
                    source=data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest  # Use nearest neighbor for categorical data
            )
                dst.write(data, 1)  # Ensure uint8 format

        print(f"Cleaned and aligned mask saved to: {final_mask}")

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



        # CHECK THE NEW MASK
        with rasterio.open(final_mask) as src:
            data = src.read(1)
            unique_values = np.unique(data)
            print(f"Unique values in mask: {unique_values}")
            nonans = nan_check(data)
            print(f'>>>nonans?={nonans}')



        # REPROJECT THE TIFS TO EPSG:4326
        print('\n>>>>>>>>>>>>>>>> reproj all layers to 4326 >>>>>>>>>>>>>>>>>')
        reprojected_image = extract_folder / f'{mask_code}_4326_image.tif'
        reprojected_dem = extract_folder / f'{mask_code}_4326_dem.tif'
        reprojected_slope = extract_folder / f'{mask_code}_4326_slope.tif'
        reprojected_mask = extract_folder / f'{mask_code}_4326_mask.tif'

        orig_images = [ image, final_dem, normalized_slope, final_mask]
        rep_images = [reprojected_image, reprojected_dem, reprojected_slope, reprojected_mask]

        for i,j in zip( orig_images, rep_images):
            # print(f'---i={i.name} j={j.name}')
            reproject_layers_to_4326_TSX(i, j)

                # CHECK THE NEW MASK
        with rasterio.open(reprojected_mask) as src:
            print(f'>>>final mask name={src.name}') 
            data = src.read(1)
            unique_values = np.unique(data)
            print(f"Unique values in reprojected mask: {unique_values}")
            nonans = nan_check(data)
            print(f'>>>nonans?={nonans}')

    # print('\n>>>>>>>>>>>>>>>> create event datacubes >>>>>>>>>>>>>>>>>')
    # create_event_datacubes_TSX(event)

