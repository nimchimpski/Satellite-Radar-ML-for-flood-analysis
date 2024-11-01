import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
import os
from pathlib import Path
import json
import time
import numpy as np



    
def check_novalues(path_to_tiff):
    with rasterio.open(path_to_tiff) as src:
        # Read the data from the first band
        band1 = src.read(1)

        # Check unique pixel values
        unique_values, counts = np.unique(band1, return_counts=True)
        print("Unique pixel values:", unique_values)
        print("Counts of each value:", counts)

def remove_nodata_from_tiff(input_tiff, output_tiff):
    """Remove the NoData flag from a TIFF, ensuring all pixel values are valid."""
    with rasterio.open(input_tiff) as src:
        # Copy the metadata and remove the 'nodata' entry
        profile = src.profile.copy()
        
        # Remove the NoData value from the profile
        if profile.get('nodata') is not None:
            print(f"Original NoData value: {profile['nodata']}")
            profile['nodata'] = None
        else:
            print("No NoData value set.")
        
        # Create a new TIFF without NoData
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            for i in range(1, src.count + 1):  # Loop through each band
                data = src.read(i)
                dst.write(data, i)
        
    print(f"Saved new TIFF without NoData to {output_tiff}")

def nan_check(input):
    if np.isnan(input).any():
        print("----Warning: NaN values found in the data.")
        return False
    else:
        print("----NO NANS FOUND")
        return True
    
def check_int16_range(dataarray):
    # TAKES A DATAARRAY NOT A DATASET
    #print("+++in small int16 range check fn+++")
    int16_min, int16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    if (dataarray < int16_min).any() or (dataarray > int16_max).any():
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        # Calculate actual min and max values in the array
        actual_min = dataarray.min().item()
        actual_max = dataarray.max().item()
        
        print(f"---Warning: Values out of int16 range found (should be between {int16_min} and {int16_max}).")
        print(f"---Minimum value found: {actual_min}")
        print(f"---Maximum value found: {actual_max}")
    
    else:
        print(f"---no exceedances int16.")

    # Optional: Replace NaN and Inf values if necessary
    # dataarray = dataarray.fillna(0)  # Replace NaN with 0 or another appropriate value
    # dataarray = dataarray.where(~np.isinf(dataarray), 0)  # Replace Inf with 0 or appropriate value
    
