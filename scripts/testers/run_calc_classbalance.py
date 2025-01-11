import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm

# Assume tiles is a list of numpy arrays (ground truth masks)
# Each tile has 1 for flooded and 0 for non-flooded pixels


def calc_ratio(tiles):
    flooded_count = 0
    non_flooded_count = 0
    for tile in tqdm(tiles.iterdir(), total=len(list(tiles.iterdir()))):
        if tile.suffix != ".tif":
            continue
        # print(f"---Processing {tile.name}")
        with rasterio.open(tile) as src:
            data = src.read(3)
            flooded_count += np.sum(data == 1)
            non_flooded_count += np.sum(data == 0)
            # print(f'---flooded_count: {flooded_count}')


    # Calculate class ratio
    total_pixels = flooded_count + non_flooded_count
    class_ratio = flooded_count / total_pixels
    # print(f'---event: {event.name}')
    print(f"{tile.parent.name} Ratio: {class_ratio:.2f}")
    return class_ratio

# train_INPUT = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\train_INPUT" )
train_INPUT = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\train_input_goodresultsmtnVNM\all_TSX_logclipmm_g_nomask1_full" )



# for event in train_INPUT.iterdir():
if True:
    
    for splitfolder in train_INPUT.iterdir():
        if not splitfolder.name in ["train", "test", "val"]:
            continue
        # if not splitfolder.name == "test":
        #     continue
        print(f"---Processing {splitfolder.name}")
        calc_ratio(splitfolder)


