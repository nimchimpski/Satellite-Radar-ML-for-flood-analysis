from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
import time
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.process_modules.process_tiffs_module import match_dem_to_mask, clip_image_to_mask_gdal
from scripts.process_modules.process_tiffs_module import calculate_and_normalize_slope, create_event_datacube_TSX,         reproject_layers_to_4326_TSX, nan_check, remove_mask_3_vals, reproject_to_4326_gdal, make_float32, make_float32_inmem
from scripts.process_modules.process_dataarrays_module import tile_datacube_rxr, calculate_global_min_max_nc, get_global_min_max
from scripts.process_modules.process_helpers import  print_tiff_info_TSX

start=time.time()

############################################################################
data_root =  Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\predictions\predict_input")
img_src =  data_root / 'image_to_run_inference_on.tif' 
min_max_file = data_root / 'TSX_process1_stats.csv'

norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
stats = None
############################################################################

            # COPY THE SAR IMAGE
            image = list(img_src.rglob('*IMAGE*.tif'))[0]

            reproj_image = img_src / '4326_image.tif'

            reproject_to_4326_gdal(image, reproj_image)

  
            final_image = img_src / 'final_image.tif'
            make_float32(reproj_image, final_image)

            print_tiff_info_TSX(image=final_image) 

            print('\n>>>>>>>>>>>>>>>> create 1 event datacube >>>>>>>>>>>>>>>>>')
            create_event_datacube_TSX(event, 'xxx')

            cube = img_src.rglob("*.nc")  

            save_tiles_path = img_src /  'prediction_tiles'"
            if save_tiles_path.exists():
                print(f"### Deleting existing tiles folder: {save_tiles_path}")
                # delete the folder and create a new one
                shutil.rmtree(save_tiles_path)
                save_tiles_path.mkdir(exist_ok=True, parents=True)

           # CALCULATE THE STATISTICS
            min_max_file = cube.parent / f'min_max.csv'
            stats = get_global_min_max(cube, 'hh', min_max_file= min_max_file)
            print(f"\n################### tiling ###################")
            # DO THE TILING AND GET THE STATISTICS
            num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask, num_nomask_pixels, num_failed_norm , num_not_256= tile_datacube_rxr(cube, save_tiles_path, tile_size=256, stride=256, norm_func=norm_func, stats=stats )
    
        print('<<<  num_tiles= ', num_tiles)
        print('<<< num_saved= ', num_saved)  
        print('<<< num_has_nans= ', num_has_nans)
        print('<<< num_novalid_layer= ', num_novalid_layer)
        print('<<< num_novalid_pixels= ', num_novalid_pixels)
        print('<<< num_nomask= ', num_nomask)
        print('<<< num_nomask_pixels= ', num_nomask_pixels)
        print('<<< num_failed_norm= ', num_failed_norm)
        print('<<< num_not_256= ', num_not_256)

        total_num_tiles += num_tiles
        total_saved += num_saved
        total_has_nans += num_has_nans
        total_novalid_layer += num_novalid_layer
        total_novalid_pixels += num_novalid_pixels
        total_nomask += num_nomask
        total_nomask_pixels += num_nomask_pixels
        total_failed_norm += num_failed_norm
        total_num_not_256 += num_not_256      
        # END OF ONE CUBE TILE PROCESSING
    # END OF ALL CUBES TILE PROCESSING  
    print(f'>>>>total num of tiles: {total_num_tiles}')
    print(f'>>>>total saved tiles: {total_saved}')
    print(f'>>>total has  NANs: {total_has_nans}')
    print(f'>>>total no valid layer : {total_novalid_layer}')
    print(f'>>>total no valid  pixels : {total_novalid_pixels}')
    print(f'>>>total no mask : {total_nomask}')
    print(f'>>>total no mask pixels : {total_nomask_pixels}')
    print(f'>>>num failed normalization : {total_failed_norm}')
    print(f'>>>num not 256: {total_num_not_256}')
    print(f'>>>all tiles tally: {total_num_tiles == total_saved + total_has_nans + total_novalid_layer + total_novalid_pixels + total_nomask + total_nomask_pixels + total_failed_norm }')

end = time.time()
# time taken in minutes to 2 decimal places
print(f"Time taken: {((end - start) / 60):.2f} minutes")

    
    


