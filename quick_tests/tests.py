import os
import rasterio
import rioxarray # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
#from tqdm import tqdm
#import cv2
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
from pathlib import Path


def main():
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")



    datas = {
        'sentinel-1-grd.tif': 'grd',
        'dem_updated.tif': 'dem',
        'slope_updated.tif': 'slope',
        'mask.tif': 'mask',
        # 'analysis_extent.tif': 'analysis_extent'
        }
    
    tile_statistics = {}

    folders = os.listdir(data_root)
    # Iterate through each folder
    for subfolder in folders:
        folder = data_root / subfolder  # Create a Path object for the folder

        # Check if the folder is a directory
        if folder.is_dir():
            # Iterate over all files in the folder
            for file in folder.iterdir():
                # Check if the file name contains '_s1_' and ends with '_img.tif'
                if 'epsg4326_sentinel12_s1_' in file.name and file.name.endswith('_img.tif'):
                    print(f"Processing file: {file}")
                    # Do something with the file, e.g., open it and process it
                
if __name__ == "__main__":
    main()