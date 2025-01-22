from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
import time
import os
import click
from rasterio.warp import calculate_default_transform, reproject, Resampling

from scripts.process_modules.process_tiffs_module import create_event_datacube_TSX, clean_mask,  reproject_to_4326_fixpx_gdal, make_float32,  create_extent_from_mask,  clip_image_to_mask_gdal

from scripts.process_modules.process_dataarrays_module import tile_datacube_rxr, compute_dataset_minmax
from scripts.process_modules.process_helpers import  print_tiff_info_TSX, write_minmax_to_json, read_minmax_from_json, compute_dataset_minmax

start=time.time()

@click.command()
@click.option('--test', is_flag=True, help='loading from test folder', show_default=False)

def main(test=None):

    ############################################################################
    # data_root = Path(r"Y:\1NEW_DATA\1data\2interim\ALL TSX PROCESSING")
    data_root = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim")

    if test:
        print('>>>TEST MODE<<<')
        dataset =  data_root / 'TSX_TO_PROCESS_TEST' 
    else:
        print('>>>FULL DATASET MODE<<<')
        dataset =  data_root / 'TSX_TO_PROCESS_INPUT' 
        # dataset=Path(r"Y:\1NEW_DATA\1data\2interim\TSX aa datacubes") 
        # dataset = Path(r'Y:\1NEW_DATA\1data\2interim\TSX aa datacubes\ok')
    make_tifs = 1
    make_datacubes = 0
    get_minmax = 0
    make_norm_tiles = 0
    norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
    minmax_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\2configs\global_minmax_INPUT\global_minmax.json")
    percent_non_flood = 100
    ############################################################################
    print(f'>>>make_tifs= {make_tifs==1} \n>>>make_datacubes= {make_datacubes==1} \n>>>get minmax= {get_minmax==1} \n>>>make_tiles= {make_norm_tiles==1}')

    # ITERATE OVER THE DATASET
    # for folder in dataset.iterdir(): # ITER ARCHIVE AND CURRENT
    if make_tifs or make_datacubes:
        # print(f'################### FOLDER={folder.name}  ###################')
        for event in tqdm(dataset.iterdir(), desc='events') : # ITER EVENT
            if event.is_dir():
                print(f'################### EVENT={event.name}  ###################')

                orig_mask = list(event.rglob('*MASK.tif'))[0]
                # GET REGION CODE FROM MASK
                mask_code = "_".join(orig_mask.name.split('_')[:2])

                # COPY  THE MASK, IMAGE, AND DEM TO THE EXTRACTED FOLDER
                if make_tifs:
                    extract_folder = event / f'{mask_code}_extracted'
                    if extract_folder.exists():
                        shutil.rmtree(extract_folder)
                    extract_folder.mkdir(exist_ok=True)

                    # COPY THE MASK
                    orig_mask = list(event.rglob('*MASK.tif'))[0]
                    # GET REGION CODE FROM MASK
                    mask_code = "_".join(orig_mask.name.split('_')[:2])

                    print(f'>>>mask_code= ',mask_code)
                    ex_mask = extract_folder / f'{mask_code}_mask.tif'
                    shutil.copy(orig_mask, ex_mask)
                    print(f'>>>mask={ex_mask.name}')

                    ex_extent = extract_folder / f'{mask_code}_extent.tif'
                    create_extent_from_mask(ex_mask, ex_extent)

                    # copy the poly
                    # poly = list(event.rglob('*POLY*.kml'))[0]
                    # ex_poly = extract_folder / f'{mask_code}_poly.tif'
                    # shutil.copy(poly, ex_poly)
                    # COPY THE SAR IMAGE
                    image = list(event.rglob('*IMAGE_*.tif') )[0]

                    print(f'>>>image={image}')
                    ex_image = extract_folder / f'{mask_code}_image.tif'
                    shutil.copy(image, ex_image)

                    # COPY THE DEM
                    # dem = list(event.rglob('*srtm*.tif'))[0]
                    # ex_dem = extract_folder / f'{mask_code}_dem.tif'
                    # print(f'>>>dem={ex_dem.name}')
                    # shutil.copy(dem, ex_dem)

                    #############################################

                    # print_tiff_info_TSX(ex_image, ex_mask)

                    #*****************************************************
                    # TODO ALL INPUTS SHOULD BE RESAMPLED TO EX. 2.5M/PX, TO MATCH THE RESAMPLING OF ALL INFERENCE INPUT IMAGES
                    #******************************************************

                    # REPROJECT THE TIFS TO EPSG:4326
                    print('\n>>>>>>>>>>>>>>>> reproj all tifs to 4326 >>>>>>>>>>>>>>>>>')
                    reproj_image = extract_folder / f'{mask_code}_4326_image.tif'
                    reproj_dem = extract_folder / f'{mask_code}_4326_dem.tif'
                    reproj_slope = extract_folder / f'{mask_code}_4326_slope.tif'
                    reproj_mask = extract_folder / f'{mask_code}_4326_mask.tif'
                    reproj_extent = extract_folder / f'{mask_code}_4326_extent.tif'

                    orig_images = [ ex_image,  ex_extent, ex_mask]
                    rep_images = [reproj_image,  reproj_extent, reproj_mask]

                    for i,j in zip( orig_images, rep_images):
                        print(f'---i={i.name} j={j.name}')
                        # check if mask or extent 
                        if 'extent' in i.name or 'mask' in i.name:
                            resampleAlg = 'near'
                        elif 'image' in i.name:
                            resampleAlg = 'bilinear'
                        print(f'---resampleAlg= {resampleAlg}')
                        reproject_to_4326_fixpx_gdal(i, j, resampleAlg, px_size=0.0001)

                    print_tiff_info_TSX(reproj_image)
                    print_tiff_info_TSX(reproj_mask)

                    # CLEAN THE MASK
                    print('\n>>>>>>>>>>>>>>>> clean mask >>>>>>>>>>>>>>>>>')
                    cleaned_mask = extract_folder / f'{mask_code}_cleaned_mask.tif'
                    clean_mask(reproj_mask, cleaned_mask)
                    
                    # print('>>>TIFF CHECK 2')
                    # print_tiff_info_TSX(image=reproj_image, mask=cleaned_mask)

                    # CLIP THE IMAGE TO THE MASK
                    print('\n>>>>>>>>>>>>>>>> clip image to mask >>>>>>>>>>>>>>>>>')
                    clipped_image = extract_folder / f'{mask_code}_clipped_image.tif'
                    clip_image_to_mask_gdal(reproj_image, cleaned_mask, clipped_image)
                    # print_tiff_info_TSX(clipped_image, cleaned_mask)


                    # MAKE FLOAT32
                    print('\n>>>>>>>>>>>>>>>> make image float32 >>>>>>>>>>>>>>>>>')
                    file_name = extract_folder / f'{mask_code}_final_image.tif'
                    final_image = make_float32(clipped_image, file_name)
                    # print_tiff_info_TSX(final_image, mask=cleaned_mask)

                    print('>>> final image= ',final_image)

                    print('\n>>>>>>>>>>>>>>>> make mask float32 >>>>>>>>>>>>>>>>>')
                    final_mask = extract_folder / f'{mask_code}_final_mask.tif'
                    make_float32(cleaned_mask, final_mask)

                    print('\n>>>>>>>>>>>>>>>> make extent float32 >>>>>>>>>>>>>>>>>')
                    final_extent = extract_folder / f'{mask_code}_final_extent.tif'
                    make_float32(reproj_extent, final_extent)

                    print_tiff_info_TSX(final_image) 
                    print_tiff_info_TSX(final_mask) 

                # CREATE AN EVENT DATA CUBE
                if make_datacubes:
                    print('\n>>>>>>>>>>>>>>>> create 1 event datacube >>>>>>>>>>>>>>>>>')
                    create_event_datacube_TSX(event, mask_code)

    # CALCULATE MIN MAX
    if get_minmax:
        if minmax_path.exists():
            print(f'>>>minmax file exists: {minmax_path}')
            # delete file
            os.remove(minmax_path)

        # else:
        print(f'>>>minmax file does not exist: getting vals')
        glob_min, glob_max = compute_dataset_minmax(dataset, band_to_read=1)
        print(f'>>>new gmin= {glob_min} gmax= {glob_max}')
        write_minmax_to_json(glob_min,glob_max, minmax_path)
        # CHECK THE NEW JSON
    minmax_data = read_minmax_from_json(minmax_path)
    print(f'>>>from json minmax now= {minmax_data}')
    minmax = (minmax_data['min'], minmax_data['max'])
    print(f'>>>minmax= {minmax}') 

    # MAKE NORMALIZED TILES
    if make_norm_tiles:
        total_num_tiles = 0
        total_saved = 0
        total_has_nans = 0
        total_novalid_layer = 0
        total_novalid_pixels = 0
        total_nomask_px = 0
        total_skip_nomask_pixels = 0
        total_failed_norm = 0
        total_num_not_256 = 0
        total_num_px_outside_extent = 0

        cubes = list(dataset.rglob("*.nc"))   
        print(f'>>>num cubes= ',len(cubes))
        for cube in tqdm(cubes, desc="### Datacubes tiled"):
            event_code = "_".join(cube.name.split('_')[:2])
            print("\n>>>>>>>>>>>> cube >>>>>>>>>>>>>>>=", cube.name)
            print(">>>event_code=", event_code)
            save_tiles_path = data_root / 'TSX_TILES' / 'NORM_TILES_FOR_SELECT_AND_SPLIT_INPUT' / f"{event_code}_normalized_tiles_{norm_func}_pcnf{percent_non_flood}"
            if save_tiles_path.exists():
                print(f"### Deleting existing tiles folder: {save_tiles_path}")
                # delete the folder and create a new one
                shutil.rmtree(save_tiles_path)
            save_tiles_path.mkdir(exist_ok=True, parents=True)


            print(f"\n################### tiling ###################")
            # DO THE TILING AND GET THE STATISTICS
            num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask_px, num_skip_nomask_pixels, num_failed_norm , num_not_256, num_px_outside_extent= tile_datacube_rxr(cube, save_tiles_path, tile_size=256, stride=256, norm_func=norm_func, stats=minmax, percent_non_flood=percent_non_flood, inference=False)

            print('<<<  num_tiles= ', num_tiles)
            print('<<< num_saved= ', num_saved)  
            print('<<< num_has_nans= ', num_has_nans)
            print('<<< num_novalid_layer= ', num_novalid_layer)
            print('<<< num_novalid_pixels= ', num_novalid_pixels)
            print('<<< num_nomask px= ', num_nomask_px)
            print('<<< num_failed_norm= ', num_failed_norm)
            print('<<< num_not_256= ', num_not_256)
            print('<<< num_px_outside_extent= ', num_px_outside_extent)

            total_num_tiles += num_tiles
            total_saved += num_saved
            total_has_nans += num_has_nans
            total_novalid_layer += num_novalid_layer
            total_novalid_pixels += num_novalid_pixels
            total_nomask_px += num_nomask_px
            total_skip_nomask_pixels += num_skip_nomask_pixels
            total_failed_norm += num_failed_norm
            total_num_not_256 += num_not_256     
            total_num_px_outside_extent += num_px_outside_extent     
            # END OF ONE CUBE TILE PROCESSING
        # END OF ALL CUBES TILE PROCESSING  
        print(f'>>>>total num of tiles: {total_num_tiles}')
        print(f'>>>>total saved tiles: {total_saved}')
        print(f'>>>total has  NANs: {total_has_nans}')
        print(f'>>>total no valid layer : {total_novalid_layer}')
        print(f'>>>total no valid  pixels : {total_novalid_pixels}')
        print(f'>>>total no mask px : {total_nomask_px}')
        print(f'>>>num failed normalization : {total_failed_norm}')
        print(f'>>>num not 256: {total_num_not_256}')
        print(f'>>>num px outside extent: {total_num_px_outside_extent}')
        print(f'>>>all tiles tally: {total_num_tiles == total_saved + total_has_nans + total_novalid_layer + total_novalid_pixels + total_skip_nomask_pixels + total_failed_norm + total_num_not_256 + total_num_px_outside_extent}')

    end = time.time()
    # time taken in minutes to 2 decimal places
    print(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == "__main__":
    main()





