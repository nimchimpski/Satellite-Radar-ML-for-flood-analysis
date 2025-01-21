

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process_modules.process_helpers import get_band_name, min_max_vals, num_band_vals, datatype_check, check_single_tile, print_tiff_info_TSX

def main():
    print('++++++++++SINGLE TILE CHECK+++++++++++++X')
    tile_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\4final\train_INPUT\TSX8_res3.25_norm3200_mt0.1_pcu0.25\train\tile_695958835_1_extracted_256_256.tif")
    # check_single_tile(tile_path)
    print_tiff_info_TSX(tile_path,1)

                
if __name__ == "__main__":
    main()
