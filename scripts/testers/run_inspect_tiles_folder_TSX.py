

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process_modules.process_helpers import get_band_name, min_max_vals, num_band_vals, datatype_check, check_single_tile

def main():
    '''
    checks tile size and pads if necessarys
    '''
    print('+++++++++++++RUNNING+++++++++++++X')
    tiles_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\predictions\predict_input_###\IMAGE_HH_tiles")
    padded = 0
    problems = 0
    folder = tiles_path






    # for folder in tiles_path.iterdir():
    if True:
        # if folder.is_dir():
        if True:
            for tile in tqdm(folder.iterdir(), total=len(list(folder.iterdir()))):
                if tile.suffix == '.tif':
                    # try:
                    if True:
                        check_single_tile(tile)

                
                    # except Exception as e:
                    #     print(f"---Error processing {tile.name}: {e}")
                        # Remove temporary file in case of error

            # print(f"Processing complete. problems: {problems} padded: {padded}.")
if __name__ == "__main__":
    main()
