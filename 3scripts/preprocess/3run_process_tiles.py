from pathlib import Path
import sys
import os
from process_tiles import process_tiles_newdir, train_split_folder
from tqdm import tqdm

# Add the parent directory of the current file to the system path
# Get the root directory and add it to sys.path
root_dir = Path(__file__).resolve().parents[2]  # Adjust level as needed to reach the root
sys.path.insert(0, str(root_dir))



def main():
    base_path = Path(r"Z:\1NEW_DATA\1data\2interim")
    # process_tiles_newdir(tiles_path, normalized_tiles_path)
    # tiles_path = base_path / "UNOSAT_FloodAI_Dataset_v2"


    # SERIOUS RUN
    normalized_tiles_path = base_path / "UNOSAT_FloodAI_Dataset_v2_norm"

    # FOR TESTS
    # normalized_tiles_path = base_path / "tests" / "tile_norm_testdata_norm"

    recursive_list = list(normalized_tiles_path.rglob('normalized_minmax_tiles'))
    for folder in  tqdm(recursive_list, desc="Processing folders"):
        
        print(f"---Processing folder: {folder.name}")
        train_split_folder(folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":
    main()