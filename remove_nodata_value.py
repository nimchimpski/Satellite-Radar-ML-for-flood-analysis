import rasterio

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

if __name__ == "__main__":
    input_tiff_path = r"Z:\1NEW_DATA\1data\2interim\tests\2022-10-02T11-16-36-flood_compressed.tif"  # Replace with your input TIFF path
    output_no_nodata_path = r"Z:\1NEW_DATA\1data\2interim\tests\nonoarse.tif"  # Replace with your output path

    # Remove the NoData flag
    remove_nodata_from_tiff(input_tiff_path, output_no_nodata_path)
