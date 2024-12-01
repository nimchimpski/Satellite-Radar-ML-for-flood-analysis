

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process_modules.process_helpers import print_tiff_info_TSX



folder_to_test = 'TSX_normalized_tiles_mt0.0_split'
train_input = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\train_input')
tiles_path = train_input / folder_to_test
padded = 0
problems = 0
for folder in tiles_path.iterdir():
    if folder.is_dir():
        tif_files = list(folder.glob("*.tif"))  # Only count `.tif` files
        for tile in tqdm(tif_files, total=len(tif_files), desc=f"Processing {folder.name}"):
            print_tiff_info_TSX(tile)
            
            # with rasterio.open(tile) as src:
            #     # Check dimensions
            #     if src.width != 256 or src.height != 256:
            #         print(f'---{tile.name} is not 256x256')
            #         problems += 1  # Increment problems counter

        print(f"Processing complete for {folder.name}. problems: {problems} padded: {padded}.")
        break

print(f"Final Summary: problems: {problems}, padded: {padded}")


