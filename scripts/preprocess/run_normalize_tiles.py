from pathlib import Path
import sys
from normalize_tiles import process_tiles_newdir
import os
from train_test_val_split import split_dataset

# Add the parent directory of the current file to the system path
# Get the root directory and add it to sys.path
root_dir = Path(__file__).resolve().parents[2]  # Adjust level as needed to reach the root
sys.path.insert(0, str(root_dir))


# Get the parent directory of the current file
# parent_dir = Path(__file__).resolve().parent[1]
# sys.path.append(str(parent_dir))

def main():
    # tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\UNOSAT_FLoodAI_Dataset_v2")
    # normalized_tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\UNOSAT_FLoodAI_Dataset_v2_minmaxnormed")
    tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\UNOSAT_FloodAI_Dataset_v2")

    normalized_tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\Flai_v2_minmaxnormed")
    
    # process_tiles_newdir(tiles_path, normalized_tiles_path)

    # to_split = Path(r'Z:\1NEW_DATA\1data\2interim\Flai_v2_minmaxnormed\FL_20200730_MMR1C48\normalized_minmax_tiles')
    to_split = Path(r'Z:\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata _\FL_20210102_MOZ3333\tiles')
    split_dataset(to_split, to_split.parent, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":
    main()