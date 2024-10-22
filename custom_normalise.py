def custom_normalize(array, lower_percentile=2, upper_percentile=98, clip_range=(0, 1)):
    """
    Normalizes a given array by scaling values between the given percentiles
    and clipping to the specified range.

    Args:
    - array: Input data (xarray.DataArray or NumPy array)
    - lower_percentile: The lower bound for normalization (default 2)
    - upper_percentile: The upper bound for normalization (default 98)
    - clip_range: Tuple specifying the range to clip the normalized data (default (0, 1))

    Returns:
    - normalized_array: The normalized data array (same type as input)
    """
    print('+++in custom_normalize fn')
    print('---array type',type(array))
    
    # Check if the input is an xarray.DataArray
    if isinstance(array, xr.DataArray):
        print('---array is xarray.DataArray')

        # Rechunk 'y' dimension into a single chunk, leave other dimensions unchanged
        array = array.chunk({'y': -1})  

        # Apply normalization while preserving metadata
        min_val = array.quantile(lower_percentile / 100.0)
        max_val = array.quantile(upper_percentile / 100.0)

        # Normalize the array
        normalized_array = (array - min_val) / (max_val - min_val)

        # Clip values to the specified range
        normalized_array = normalized_array.clip(min=clip_range[0], max=clip_range[1])
        normalized_array = normalized_array.astype('float32')
        return normalized_array

    else:
        print('---array is not xarray.DataArray')
        # Handle as a NumPy array if not xarray.DataArray
        min_val = np.nanpercentile(array, lower_percentile)
        max_val = np.nanpercentile(array, upper_percentile)
        
        # Normalize the array
        normalized_array = (array - min_val) / (max_val - min_val)
        
        # Clip values to the specified range
        normalized_array = np.clip(normalized_array, clip_range[0], clip_range[1])

        return normalized_array