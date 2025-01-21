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
import click
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.train_modules.train_classes import UnetModel
from scripts.process_modules.process_tiffs_module import  create_event_datacube_TSX_inf,reproject_to_4326_gdal, make_float32_inf, resample_tiff_gdal
from scripts.process_modules.process_dataarrays_module import tile_datacube_rxr
from scripts.process_modules.process_helpers import  print_tiff_info_TSX, check_single_input_filetype, rasterize_kml_rasterio, compute_image_minmax, process_raster_minmax, path_not_exists, read_minmax_from_json, normalize_imagedata_inf, read_raster, write_raster
from collections import OrderedDict
from skimage.morphology import binary_erosion

start=time.time()

def make_prediction_tiles(tile_folder, metadata, model, device, threshold):
    predictions_folder = Path(tile_folder).parent / f'{tile_folder.stem}_predictions'
    if predictions_folder.exists():
        print(f"--- Deleting existing predictions folder: {predictions_folder}")
        # delete the folder and create a new one
        shutil.rmtree(predictions_folder)
    predictions_folder.mkdir(exist_ok=True)

    for tile_info in tqdm(metadata, desc="Making predictions"):
        tile_path = tile_folder /  tile_info["tile_name"]
        pred_path = predictions_folder / tile_info["tile_name"]

        with rasterio.open(tile_path) as src:
            tile = src.read(1).astype(np.float32)  # Read the first band
            profile = src.profile   
            nodata_mask = src.read_masks(1) == 0  # True where no-data

        # Prepare tile for model
        tile_tensor = torch.tensor(tile).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims

        # Perform inference
        with torch.no_grad():
            pred = model(tile_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # Convert logits to probabilities
            pred = (pred > threshold).astype(np.float32)  # Convert probabilities to binary mask
            pred[nodata_mask] = 0  # Mask out no-data areas

        # Save prediction as GeoTIFF
        profile.update(dtype=rasterio.float32)
        with rasterio.open(pred_path, "w", **profile) as dst:
            dst.write(pred.astype(np.float32), 1)

    return predictions_folder



def stitch_tiles(metadata, prediction_tiles, save_path, image):
    ''''
    metadata =list
    '''
    # GET CRS AND TRANSFORM
    with rasterio.open(image) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        print('>>>src shape:',src.shape)
    
        # INITIALIZE THE STITCHED IMAGE AND COUNT
        # give stitched_image the same crs, transform and shape as the source image
        stitched_image = np.zeros((height, width))
        # print(">>>stitched_image dtype:", stitched_image.dtype)
        print(">>>stitched_image shape:", stitched_image.shape)
        #print unique values in the stitched image
        # print(f'>>>unique values in empty stitched image: {np.unique(stitched_image)}')

    for tile_info in tqdm(metadata, desc="Stitching tiles"):
        tile_name = tile_info["tile_name"]
        # Extract position info from metadata
        x_start, x_end = tile_info["x_start"], tile_info["x_end"]
        y_start, y_end = tile_info["y_start"], tile_info["y_end"]

        # Find the corresponding prediction tile
        tile = prediction_tiles / tile_name

        # Load the tile
        with rasterio.open(tile) as src:
            tile = src.read(1).astype(np.float32)
            # Debugging: Print tile info and shapes
            # print(f">>>Tile shape: {tile.shape}")
        # print(f">>> Tile info: {tile_info}")

        # Extract the relevant slice from the stitched image
        stitched_slice = stitched_image[y_start:y_end, x_start:x_end]
        if (stitched_slice.shape[0] == 0) or (stitched_slice.shape[0] == 1):
            continue
        
        # Validate dimensions
        if stitched_slice.shape != tile.shape:
            if (stitched_slice.shape[0] == 0) or (stitched_slice.shape[1] == 0):
                continue
            print(f"---Mismatch: Stitched slice shape: {stitched_slice.shape}, ---Tile shape: {tile.shape}")
            slice_height, slice_width = stitched_slice.shape
            tile = tile[:slice_height, :slice_width]  # Crop tile to match slice
            # Debugging: Print the new tile shape
            print(f">>>New tile shape: {tile.shape}")


        # Add the tile to the corresponding position in the stitched image
        stitched_image[y_start:y_end, x_start:x_end] += tile
        # PRINT STITCHED IMAGE SIZE
        # print(f">>>Stitched image shape: {stitched_image.shape}")
    print(f'---crs: {crs}')
    # Save the stitched image as tif, as save_path
    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=stitched_image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(stitched_image, 1)
    # with rasterio.open(save_path) as src:
    #     print("No-data value:", src.nodata)
        
    return stitched_image


def clean_checkpoint_keys(state_dict):
    """Fix the keys in the checkpoint by removing extra prefixes."""
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.")
        elif key.startswith("model."):
            new_key = key.replace("model.", "")
        else:
            new_key = key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict

@click.command()
@click.option('--test', is_flag=True, help='loading from test folder', show_default=False)
def main(test=None):
    with open(Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\2configs\floodaiv2_config.json")) as file:
        config = json.load(file)
    
    # print(f'>>>config: {config}')
    
    threshold = config["threshold"] # PREDICTION CONFIDENCE THRESHOLD
    img_src = Path(config["input_folder"])
    print(f'>>>img source= {img_src}')
    output_filename = config["output_filename"]

    print(f'>>>threshold: {threshold}')
    if test:
        print("TEST SOURCE")
        img_src = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\4final\predict_input_test")

    print(f'>>>img_src: {img_src}')

    if path_not_exists(img_src):
        print(f"---No input folder found in {img_src}")
        return
    
    save_path = img_src / f'{output_filename}_th{threshold}_WATER.tif'
    if save_path.exists():
        try:
            print(f"--- Deleting existing prediction file: {save_path}")
            save_path.unlink()
        except Exception as e:
            print(f"--- Error deleting existing prediction file: {e}")

    ############################################################################
    minmax_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\2configs\global_minmax_INPUT\global_minmax.json")
    if path_not_exists(minmax_path):
        return
    norm_func = 'logclipmm_g' # 'mm' or 'logclipmm'
    stats = None
    # ckpt = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\4results\checkpoints\good\mtnweighted_NO341_3__BS16__EP10_weighted_bce.ckpt")
    ckpt_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\5checkpoints\ckpt_INPUT")

    ############################################################################
    print(f'>>> CHECK LAYERDICT NAMES=FILENAMES IN FOLDER <<<')
    # FIND THE CKPT
    ckpt = next(ckpt_path.rglob("*.ckpt"), None)
    print(f'>>> threshold: {threshold}')
    if ckpt is None:
        print(f"---No checkpoint found in {ckpt_path}")
        return
    print(f'>>>ckpt: {ckpt.name}')

    # FIND THE SAR IMAGE
    image = check_single_input_filetype(img_src, 'image', '.tif')
    if image is None:
        return
    # print(f'>>>image: {image}')
    # poly = check_single_input_filetype(img_src,  'poly', '.kml')
    # if poly is None:
        # return

    # GET REGION CODE FROM MASK TODO
    image_code = "_".join(image.name.split('_')[:2])
    print(f'>>>image_code= ',image_code)

    if True:
    
        # CREATE THE EXTRACTED FOLDER
        extracted = img_src / f'{image_code}_extracted'
        if extracted.exists():
            # print(f"--- Deleting existing extracted folder: {extracted}")
            # delete the folder and create a new one
            shutil.rmtree(extracted)
        extracted.mkdir(exist_ok=True)

        # CHANGE DATATYPE TO FLOAT32
        print('>>>CHANGING DATATYPE')
        image_32 = extracted / f'{image_code}_32.tif'
        make_float32_inf(image, image_32)
        print_tiff_info_TSX(image_32, 1)

        # RESAMPLE TO 2.5
        print('>>>RESAMPLING')
        resamp_image = extracted / f'{image_32.stem}_resamp'
        resample_tiff_gdal(image_32, resamp_image, target_res=2.5)
        print_tiff_info_TSX(resamp_image, 2)

        # with rasterio.open(image) as src:
            # print(f'>>>src shape= ',src.shape)

        # SORT OUT ANALYSIS EXTENT

        # ex_extent = extracted / f'{image_code}_extent.tif'
        # create_extent_from_mask(image, ex_extent)
        # rasterize_kml_rasterio( poly, ex_extent, pixel_size=0.0001, burn_value=1)

        # REPROJECT IMAGE
        print('>>>REPROJECTING')
        final_image = extracted / 'final_image.tif'
        reproject_to_4326_gdal(resamp_image, final_image, resampleAlg = 'bilinear')
        print_tiff_info_TSX(final_image, 3)

        # reproj_extent = extracted / f'{image_code}_4326_extent.tif'
        # reproject_to_4326_gdal(ex_extent, reproj_extent)
        # fnal_extent = extracted / f'{image_code}_32_final_extent.tif'
        # make_float32_inf(reproj_extent, final_extent

    extracted = img_src / f'{image_code}_extracted'
    final_image = extracted / 'final_image.tif'


    # GET THE TRAINING MIN MAX STATS
    statsdict =  read_minmax_from_json(minmax_path)
    stats = (statsdict["min"], statsdict["max"])



    if True:
        create_event_datacube_TSX_inf(img_src, image_code)

    cube = next(img_src.rglob("*.nc"), None)  
    save_tiles_path = img_src /  f'{image_code}_tiles'

    if save_tiles_path.exists():
        # print(f">>> Deleting existing tiles folder: {save_tiles_path}")
        # delete the folder and create a new one
        shutil.rmtree(save_tiles_path)
        save_tiles_path.mkdir(exist_ok=True, parents=True)
        # CALCULATE THE STATISTICS

    # DO THE TILING
    tiles, metadata = tile_datacube_rxr(cube, save_tiles_path, tile_size=256, stride=256, norm_func=norm_func, stats=stats, percent_non_flood=0, inference=True) 
    # print(f">>>{len(tiles)} tiles saved to {save_tiles_path}")
    # print(f">>>{len(metadata)} metadata saved to {save_tiles_path}")
    # metadata = Path(save_tiles_path) / 'tile_metadata.json'


    # INITIALIZE THE MODEL
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel( encoder_name="resnet34", in_channels=1, classes=1, pretrained=False 
    )   
    model.to(device)
    # LOAD THE CHECKPOINT
    ckpt_path = Path(ckpt)
    checkpoint = torch.load(ckpt_path)

    cleaned_state_dict = clean_checkpoint_keys(checkpoint["state_dict"])


    # EXTRACT THE MODEL STATE DICT
    # state_dict = checkpoint["state_dict"]

    # LOAD THE MODEL STATE DICT
    model.load_state_dict(cleaned_state_dict)

    # SET THE MODEL TO EVALUATION MODE
    model.eval()

    prediction_tiles = make_prediction_tiles(save_tiles_path, metadata, model, device, threshold)

    # STITCH PREDICTION TILES
    prediction_img = stitch_tiles(metadata, prediction_tiles, save_path, final_image)
    # print prediction_img size
    # print(f'>>>prediction_img shape:',prediction_img.shape)
    # display the prediction mask
    plt.imshow(prediction_img, cmap='gray')
    # plt.show()



    end = time.time()
    # time taken in minutes to 2 decimal places
    print(f"Time taken: {((end - start) / 60):.2f} minutes")

if __name__ == "__main__":
    main()