

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files
from scripts.process_modules.process_helpers import get_band_name, min_max_vals, num_band_vals, datatype_check

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
                        with rasterio.open(tile) as src:
                            # print('---tile:', tile.name)
                            # Read all datasdsdfsdfsdvsdvs
                            data = src.read()
                            # LOOP THRU BANDS
                            for band in range(1, src.count + 1):
                                band_data = data[band - 1]
                                name = get_band_name(band, src)
                                print(f'\n---{band}={name}')
                                numvals =  num_band_vals(band_data)

                                if name in ['mask', 'extent']:

                                    # CHECK NUM UNIQUE VALS
                                    if numvals >2:
                                        print('---not 2 vals , ', numvals)
                                        return
                                      # CHECK VALS ARE 0 OR 1
                                    min, max = min_max_vals(band_data)
                                    if round(min) not in [0, 1] or round(max) not in [0, 1]:
                                        print(f'---min={min}, max={min}')
                                        pass
                                else:
                                    # print(f'--num_band_vals={numvals}')

                                    # CHECK MIN MAX INSIDE 0 AND 1 - NORMALIZED
                                    min, max = min_max_vals(band_data)
                                    if min == max:
                                        print(f'---uniform values in {name} band: {min}, {max}')
                                    if min < 0 or max > 1:
                                        print(f'---out of range values in {name} band: {min}, {max}')
                                        raise ValueError(f'---out of range values in {name} band: {min}, {max}')



                            
'''
     
                                            problems += 1

                                        # Normalize uniform DEM bands to 0.5
                                        if band_name == 'dem':
                                            if (np.min(band_data) == np.max(band_data)) and (np.min(band_data) < 0 or np.max(band_data) > 1):
                                                print('---out of range')
                                                print(f"Normalizing uniform DEM band in {tile.name} with value {np.min(band_data)}")
                                                data[band - 1] = np.full_like(band_data, 0.5)

                                    # ENSURE MASK IS 0 OR 1
                                    if band_name in ['mask', 'valid']:
                                        unique_values = np.unique(band_data)
                                        if not set(unique_values).issubset({0, 1}):
                                            print(f"---Invalid mask in {tile.name} band {band}: min={np.min(band_data)}, max={np.max(band_data)}")

                                    # Write the band back to the destination file
                                    # dst.write(data[band - 1], band)

                                    # Retain band descriptions
                                    # if band_name:
                                        # dst.set_band_description(band, band_name)

                            # Replace original file with modified temporary file
                            # shutil.move(temp_tile, tile)
'''
                    # except Exception as e:
                    #     print(f"---Error processing {tile.name}: {e}")
                        # Remove temporary file in case of error

            # print(f"Processing complete. problems: {problems} padded: {padded}.")
if __name__ == "__main__":
    main()
