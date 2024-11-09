from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
import numpy as np
import rasterio
from normalize_tiles import custom_normalize, normalise_a_tile, process_tiles_newdir
import sys


# Get the parent directory of the current file
# parent_dir = Path(__file__).resolve().parent[1]
# sys.path.append(str(parent_dir))

def main():
    # tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\UNOSAT_FLoodAI_Dataset_v2")
    # normalized_tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\UNOSAT_FLoodAI_Dataset_v2_minmaxnormed")
    tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata")
    normalized_tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata_minmaxnormed")
    
    process_tiles_newdir(tiles_path, normalized_tiles_path)

if __name__ == "__main__":
    main()