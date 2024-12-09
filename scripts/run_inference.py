import torch
from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
import time
import os
import xarray as xr
import json
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.process_modules.process_tiffs_module import match_dem_to_mask, clip_image_to_mask_gdal
from scripts.process_modules.process_tiffs_module import calculate_and_normalize_slope, create_event_datacube_TSX_inf,         reproject_layers_to_4326_TSX, nan_check, remove_mask_3_vals, reproject_to_4326_gdal, make_float32_inf, make_float32_inmem
from scripts.process_modules.process_dataarrays_module import tile_datacube_rxr, calculate_global_min_max_nc, get_global_min_max
from scripts.process_modules.process_helpers import  print_tiff_info_TSX

start=time.time()

def make_prediction_tiles(tile_folder, metadata, model, device):
    predictions_folder = Path(tile_folder).parent / f'{tile_folder.stem}_prediction'

    for tile_info in metadata:
        tile_path = os.path.join(tile_folder, tile_info["tile_name"])
        pred_path = os.path.join(predictions_folder, tile_info["tile_name"])

        with rasterio.open(tile_path) as src:
            tile = src.read(1).astype(np.float32)  # Read the first band
            profile = src.profile   

        # Prepare tile for model
        tile_tensor = torch.tensor(tile).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims

        # Perform inference
        with torch.no_grad():
            pred = model(tile_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # Convert logits to probabilities

        # Save prediction as GeoTIFF
        profile.update(dtype=rasterio.float32)
        with rasterio.open(pred_path, "w", **profile) as dst:
            dst.write(pred.astype(np.float32), 1)

    return predictions_folder



def stitch_tiles(metadata_path, tile_folder, output_path, image):

    # LOAD METADATA
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # GET CRS AND TRANSFORM
    with rasterio.open(image) as src:
        crs = src.crs
        transform = src.transform


    # INITIALIZE THE STITCHED IMAGE AND COUNT
    stitched_image = np.zeros(image.shape)
    count = np.zeros(image.shape)

    # READ THE TILES FROM METADATA
    for tile_info in metadata:
        tile = tile_folder / tile_info["tile_name"]

        x_start, y_start = tile_info["x_start"], tile_info["y_start"]
        x_end, y_end = tile_info["x_end"], tile_info["y_end"]

        stitched_image[y_start:y_end, x_start:x_end] += tile
        count[y_start:y_end, x_start:x_end] += 1

    # Average overlapping regions
    stitched_image = np.divide(stitched_image, count, where=count > 0)
    # Save the stitched image
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=stitched_image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(stitched_image, 1)

    print(f"---Stitched image saved to {output_path}")
    return stitched_image

############################################################################
img_src =  Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\predictions\predict_input")
min_max_file = img_src / 'TSX_process1_stats.csv'
norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
stats = None
model = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\4results\checkpoints\TSX_logclipmm_g_mt0.3__BS16__EP10_WEIGHTED_BCE")
############################################################################

def main():
    # FIND THE SAR IMAGE
    image = list(img_src.rglob('*image*.tif'))[0]
    print(f'>>>image.name = ',image.name)

    # # GET REGION CODE FROM MASK
    image_code = "_".join(image.name.split('_')[:2])
    print(f'>>>image_code= ',image_code)

    # # CREATE THE EXTRACTED FOLDER
    extracted = image.parent / f'{image_code}_extracted'
    extracted.mkdir(exist_ok=True)

    reproj_image = extracted / f'{image_code}_4326.tif'
    reproject_to_4326_gdal(image, reproj_image)

    final_image = extracted / f'{reproj_image.stem}_32_final_image.tif'
    make_float32_inf(reproj_image, final_image)


    print_tiff_info_TSX(image=final_image) 
# 
    # print('\n>>>>>>>>>>>>>>>> create 1 event datacube >>>>>>>>>>>>>>>>>')
    create_event_datacube_TSX_inf(img_src, image_code)

    cube = next(img_src.rglob("*.nc"), None)  
    save_tiles_path = img_src /  f'{image_code}_tiles'
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
    tiles, metadata = tile_datacube_rxr(cube, save_tiles_path, tile_size=256, stride=256, norm_func=norm_func, stats=stats, inference=True) 
    print(f"---{len(tiles)} tiles saved to {save_tiles_path}")
    print(f"---{len(metadata)} metadata saved to {save_tiles_path}")
    metadata = Path(save_tiles_path) / 'metadata.json'
    # MAKE PREDICTION TILES
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model, map_location=device)

    prediction_tiles = make_prediction_tiles(save_tiles_path, metadata, model, device)

    # STITCH PREDICTION TILES
    prediction_img = stitch_tiles(metadata, prediction_tiles, image)

    # display the prediction mask
    plt.imshow(prediction_img, cmap='gray')
    plt.show()

    print(f"---Prediction mask saved to {prediction_img}")


    end = time.time()
    # time taken in minutes to 2 decimal places
    print(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == "__main__":
    main()