import rasterio
from pathlib import Path
import numpy as np
import os


def main():
    # tiles_path = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata\FL_20210102_MOZ3333/tiles")
    tiles_path = Path(r"1data/2interim/Flai_v2_minmaxnormed/FL_20200730_MMR1C48/normalized_minmax_tiles")

    for tile in tiles_path.iterdir():
        print('---tile:', tile)
        try:
            with rasterio.open(tile) as src:
                for band in range(1, src.count + 1):
                    if band:
                        band_name = src.descriptions[band - 1]
                        if band_name:
                            band_name = src.descriptions[band - 1].lower()
                            print('---band' ,band,  band_name)
                            data = src.read(band)
                            print('---min:', np.min(data),'max:', np.max(data))

                        else:
                            print('---no band name')
                    else:
                        print('---no band')
                    

        except rasterio.errors.RasterioIOError as e:
            print(f"eeeError opening file {tile}: {e}")
        except PermissionError as e:
            print("pppPermission denied for file {tile}: {e}")
        return

if __name__ == "__main__":
    main()
