import os
import rasterio
import rioxarray # very important!
import numpy as np
import xarray as xr
from rasterio.enums import ColorInterp
from pathlib import Path
# from WORKING_datacube_from_local import check_nan_gdal
# from WORKING_datacube_from_local import fill_nodata_with_zero
# from WORKING_datacube_from_local import create_vv_and_vh_tifs
from check_int16_exceedance2 import check_int16_exceedance2

def main():
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")

    datacube_path = data_root / 'datacube_vWebb1.nc'

        # Check if the file exists
    if not datacube_path.exists():
        print(f"Error: {datacube_path} does not exist.")
        return

    # Load the NetCDF file using xarray
    try:
        datacube = xr.open_dataset(datacube_path)
        print(f"Successfully loaded datacube from {datacube_path}")
    except Exception as e:
        print(f"Error loading datacube: {e}")
        return
    
    check_int16_exceedance2(datacube)


                
if __name__ == "__main__":
    main()