import rasterio

# Open the TIFF file
with rasterio.open('dataset_renametest/1/sentinel12_s1_1_img.tif') as dataset:
    # Get metadata
    metadata = dataset.tags()

    # Print metadata
    print(dataset.profile)
    
    # Read the first band
    band1 = dataset.read(1)
    print(band1)  # Print the raster data (as a numpy array)

    # Print the dimensions of the raster
    print(f"Width: {dataset.width}, Height: {dataset.height}")

    # Access the coordinate reference system (CRS)
    print(f"CRS: {dataset.crs}")

    # Access bounding box
    print(f"Bounds: {dataset.bounds}")