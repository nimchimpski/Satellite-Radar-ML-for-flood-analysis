import xarray as xr

# Define the threshold for excess values, e.g., int16 limits (-32768 to 32767)


def check_int16_exceedance(datacube):
    """
    Check for values exceeding the int16 limits in a dataset or data array.
    Can handle both sub-datacubes (single events) and multi-event datacubes.
    """
    def find_exceedances(data_array, layer_name):
        """
        Check a DataArray for values exceeding the int16 limits and print their locations.
        """
        INT16_MIN = -32768
        INT16_MAX = 32767
        # Find where values exceed int16 limits
        exceed_mask = (data_array < INT16_MIN) | (data_array > INT16_MAX)
        exceed_mask = exceed_mask.compute()  # Force Dask to compute the result
    
        if exceed_mask.any():
            # If there are any exceedances, print or return their locations
            exceed_indices = exceed_mask.nonzero()  # Get the indices where the condition is true
            print(f"Exceedances found in {layer_name} at locations: {exceed_indices}")
        else:
            print(f"No exceedances found in {layer_name}.")


    if isinstance(datacube, xr.Dataset):
        print(f"Checking Dataset with {len(datacube.data_vars)} variables")
        # If it's a Dataset, loop through all the layers/variables
        for layer_name, data_array in datacube.data_vars.items():
            print(f"Checking layer: {layer_name}")
            find_exceedances(data_array, layer_name)
    elif isinstance(datacube, xr.DataArray):
        print(f"Checking single DataArray")
        # If it's a DataArray, check the single array
        find_exceedances(datacube, "single layer")
    else:
        raise TypeError("Expected xarray.Dataset or xarray.DataArray, got something else.")




