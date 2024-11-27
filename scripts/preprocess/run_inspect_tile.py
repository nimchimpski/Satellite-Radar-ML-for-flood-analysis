# import rasterio
# from pathlib import Path
# import numpy as np
# import os
# from tqdm import tqdm


# def main():
#     # tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata\FL_20210102_MOZ3333/tiles")
#     tiles_path = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\tests\mask_check')
#     counter = 0
#     padded = 0
#     for tile in tqdm(tiles_path.iterdir(), total=len(list(tiles_path.iterdir()))):
#         # print('---tile:', tile.name)
#         try:
#             with rasterio.open(tile) as src:
#                 # print('---shape:', src.shape)
#                 # OPEN THE FILE TO WRITE TO
#                 with rasterio.open(tile, 'w', driver='GTiff', height=src.height, width=src.width, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=src.transform) as dst:

#                     # IF NOT 256X256 ADD REFLECT PADDING
#                     if src.width != 256 or src.height != 256:
#                         print('---padding')
#                         pad_width = 256 - src.width
#                         pad_height = 256 - src.height
#                         data = np.pad(src.read(), ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
#                         dst.write(data)
#                         padded += 1
#                     else:
#                         data = src.read()
#                         # print('---min:', np.min(data),'max:', np.max(data))
#                     # print('---src:', src.count)
#                     for band in range(1, src.count + 1):
#                         if band:
#                             band_name = src.descriptions[band - 1]
#                             if band_name:
#                                 band_name = src.descriptions[band - 1].lower()
#                                 data = src.read(band)

#                                 # IF NON MASKS HAVE EQUAL MIN AND MAX, set to 0.5
#                                 if np.min(data) == np.max(data) and band_name == 'dem':
#                                     print(f"Normalizing uniform DEM band with value {np.min(data)}")
#                                     data = np.full_like(data, 0.5)  # Replace with a constant neutral value (0.5)
#                                     dst.write(data, band)


#                                 # MAX AND MIN MUST BE 0 AND 1
#                                 if (np.min(data) not in [0, 1]) or  (np.max(data) not in [0, 1]):
#                                     print('---band:', band_name)
#                                     print('---min:', np.min(data),'max:', np.max(data))
#                                 # PRINT UNIQUE VALUES
#                                 # if band_name == 'mask':
#                                     # print('---mask')
#                                     # unique, counts = np.unique(data, return_counts=True)
#                                     # print('---unique:', unique, 'counts:', counts)

#                             else:
#                                 print('---no band name')
#                         else:
#                             print('---no band')
        # except rasterio.errors.RasterioIOError as e:
        #     print(f"eeeError opening file {tile}: {e}")
        # except PermissionError as e:
        #     print("pppPermission denied for file {tile}: {e}")

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files


def main():
    tiles_path = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\train_input\UNOSAT_FloodAI_Dataset_v2_norm_px0.0_split')
    padded = 0
    problems = 0

    for folder in tiles_path.iterdir():
        if folder.is_dir():
            for tile in tqdm(folder.iterdir(), total=len(list(folder.iterdir()))):
                if tile.suffix == '.tif':
                    try:
                        with rasterio.open(tile) as src:
                            # Create a temporary file for writing
                            temp_tile = tile.parent / f"temp_{tile.name}"

                            with rasterio.open(temp_tile, 'w', driver='GTiff',
                                               height=256, width=256, count=src.count,
                                               dtype=src.dtypes[0], crs=src.crs, transform=src.transform) as dst:

                                # Read all data
                                data = src.read()

                                # Apply reflect padding if dimensions are not 256x256
                                if src.width != 256 or src.height != 256:
                                    pad_width = max(256 - src.width, 0)
                                    pad_height = max(256 - src.height, 0)
                                    print(f"---Padding {tile.name}")
                                    data = np.pad(data, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
                                    padded += 1

                                # Process each band
                                for band in range(1, src.count + 1):
                                    band_data= data[band - 1]
                                    band_name = src.descriptions[band - 1].lower() if src.descriptions[band - 1] else None
                                    # print('---band:', band_name)

                                    # IF BAND IS UNIFORM 0.5 IT PASSES
                                    if (np.min(band_data) == np.max(band_data) == 0.5):
                                        print(f'---band: {band_name} is unbiform and set to 0.5')
                                    else:
                                        # IF UNIQUE VALS NOT 0 OR 1 - FLAG IT
                                        if ((np.min(band_data) not in [0,  1]) or (np.max(band_data) not in [0, 1])):
                                            print(f"---Band {band} in {tile.name} has values outside [0, 1]: min={np.min(band_data)}, max={np.max(band_data)}")
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
                                    dst.write(data[band - 1], band)

                                    # Retain band descriptions
                                    if band_name:
                                        dst.set_band_description(band, band_name)

                            # Replace original file with modified temporary file
                            shutil.move(temp_tile, tile)

                    except Exception as e:
                        print(f"---Error processing {tile.name}: {e}")
                        # Remove temporary file in case of error
                        temp_tile.unlink(missing_ok=True)

            # print(f"Processing complete. problems: {problems} padded: {padded}.")





if __name__ == "__main__":
    main()
