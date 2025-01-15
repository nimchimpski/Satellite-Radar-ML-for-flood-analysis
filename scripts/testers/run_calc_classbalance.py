import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm

# Assume tiles is a list of numpy arrays (ground truth masks)
# Each tile has 1 for flooded and 0 for non-flooded pixels


def calc_ratio(tiles):
    flooded_count = 0
    non_flooded_count = 0
    tileslist = list(tiles.iterdir())
    print(f'---{tiles.name} folder = {len(tileslist)} files')
    for tile in tqdm(tiles.iterdir(), total=len(tileslist), desc="TILES"):
        if not tile.suffix == '.tif':
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

# train_INPUT = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\4final\train_INPUT\695971749_1.nc_normalized_tiles_logclipmm_g_pcnf100_mt0.3_pcu0.0" )
train_INPUT = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\4final\train_INPUT" )



# for event in train_INPUT.iterdir():
# if True:
for folder in train_INPUT.iterdir():
    for splitfolder in folder.iterdir():        
        if splitfolder.is_dir and  splitfolder.name in ["train", "test", "val"]:
            # if not splitfolder.name == "test":
            #     continue
            # print(f"---Processing {splitfolder.name}")
            calc_ratio(splitfolder)
        elif splitfolder.suffix == '.txt':
            # print(f"---Processing txt {splitfolder.name}")
            with open(splitfolder, 'r') as f:
                lines = f.readlines()
                numlines = len(lines)
                print(f"---{splitfolder.name} has {numlines} lines")


