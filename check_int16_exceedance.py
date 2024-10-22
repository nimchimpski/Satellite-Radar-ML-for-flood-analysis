import xarray as xr
from pathlib import Path

def check_int16_exceedance(datacube_input):
    print('++++in check_int16_exceedance')
    # Load the datacube (assuming it's a NetCDF or Zarr format)
    if isinstance(datacube_input, Path):
        print(f"---Loading dataset from file: {datacube_input}")

        datacube = xr.open_dataset(datacube_input)  # Use xr.open_zarr(datacube_path) if in Zarr format
    else:
        print(f"---Loading dataset from in memory object")
        datacube = datacube_input
    # Define the valid int16 range
    int16_min = -32768
    int16_max = 32767
    
    # Loop through each layer in the datacube
    for layer_name in datacube.data_vars:  # Assuming each layer is a separate variable
        print(f"---Checking layer: {layer_name}")
        layer = datacube[layer_name]
        
        # Find where values are outside the int16 range
        invalid_values = ((layer < int16_min) | (layer > int16_max))
        
        # Count the number of invalid values
        count_invalid = invalid_values.sum().item()
        
        if count_invalid > 0:
            print(f"---Layer '{layer_name}' has {count_invalid} values exceeding int16 limits.")
            
            # Optionally, print the indices where invalid values are located
            invalid_indices = layer.where(invalid_values, drop=True)
            print(f"---Excessive values are located at:\n{invalid_indices}")
        else:
            print(f"---Layer '{layer_name}' has no values exceeding int16 limits.")
    
    datacube.close()

