import rasterio
from rasterio import windows

# Define the size of each tile (e.g., 256x256 pixels)
tile_size = 256

# Open the dataset (assumed to be in UTM)
with rasterio.open('input_utm.tif') as dataset:
    # Get the width and height of the dataset
    width = dataset.width
    height = dataset.height
    
    # Loop over the dataset to create tiles
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            # Define the window for the current tile
            window = windows.Window(i, j, tile_size, tile_size)
            
            # Read the tile data
            tile_data = dataset.read(window=window)
            
            # Save the tile in the same CRS (UTM)
            transform = windows.transform(window, dataset.transform)
            
            # Write each tile to a new file
            with rasterio.open(f'tile_{i}_{j}.tif', 'w', driver='GTiff',
                               height=tile_size, width=tile_size,
                               count=dataset.count, dtype=tile_data.dtype,
                               crs=dataset.crs, transform=transform) as dst:
                dst.write(tile_data)
