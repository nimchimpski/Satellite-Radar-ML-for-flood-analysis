import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import ColorInterp


def tile_to_dir(stack, path, tile_size, stride, event):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    - bucket(str): AWS S3 bucket to write tiles to
    """
    print(f"Writing tempfiles to {path}")
    os.makedirs(path, exist_ok=True)

    # Calculate the number of full tiles in x and y directions
    # num_x_tiles = stack[0].x.size // tile_size
    # num_y_tiles = stack[0].y.size // tile_size
    num_x_tiles = max(stack[0].x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(stack[0].y.size + stride - 1, 0) // stride + 1

    counter = 0
    counter_valid = 0
    counter_nomask = 0
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            # Calculate the start and end indices for x and y dimensions
            # for the current tile
            # x_start = x_idx * tile_size
            # y_start = y_idx * tile_size
            # x_end = min(stride * x_idx + tile_size, stack[0].x.size)
            # y_end = min(stride * y_idx + tile_size, stack[0].y.size)

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, stack[0].x.size)
            y_end = min(y_start + tile_size, stack[0].y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            parts = [part['band_data'][:, y_start:y_end, x_start:x_end] for part in stack]

            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 250 == 0:
                print("Counted {} tiles".format(counter))

            # set the criterion here.
            if not filter_nodata(tile):
                continue

            if not filter_noanalysis(tile):
                continue

            if not filter_nomask(tile):
                counter_nomask +=1
                continue

            # if (tile.sel(band='mask').sum().values.tolist() == 0):
            #     import pdb; pdb.set_trace()

            # Track band names and color interpretation
            tile.attrs["long_name"] = [str(x.values) for x in tile.band]
            color = [ColorInterp.blue, ColorInterp.green, ColorInterp.red] + [
                ColorInterp.gray
            ] * (len(tile.band) - 3)

            name = os.path.join(path, "tile_{event}_{version}_{x_idx}_{y_idx}.tif".format(
                path=path,
                event=event,
                version=VERSION,
                x_idx=x_idx,
                y_idx=y_idx
            ))
            tile.rio.to_raster(name, compress="deflate", nodata=np.nan)
            counter_valid += 1

            with rasterio.open(name, "r+") as rst:
                rst.colorinterp = color
                # rst.update_tags(date=VERSION)

    return counter, counter_valid, counter_nomask