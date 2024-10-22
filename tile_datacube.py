import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import ColorInterp
import rioxarray as rxr 
from pathlib import Path


def tile_to_dir(datacube, save_path, tile_size=256, stride, event):
    """
    Function to tile a multi-dimensional imagery datacube while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - datacube (xarray.Dataset): The input multi-dimensional imagery datacube.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    """

    def filter_nodata(tile):
        """
        Filter tiles based on no-data pixels.
        Args:
        - tile (xarray.Dataset): A subset of data representing a tile.
        Returns:
        - bool: True if the tile is approved, False if rejected.
        """
        bands_to_check = ["dem", "slope", "vv", "vh"]
        for band in bands_to_check:
            if int(np.isnan(tile.sel(band=band)).sum()):
                return False
        return True  # If both conditions pass
    
    def filter_nomask(tile):
        if (tile.sel(band='mask').sum().values.tolist() == 0):
            return False        
        return True  # If both conditions pass

    def filter_noanalysis(tile):
        '''
        Filters out tiles without analysis extent (in this case, using valid.tiff as analysis_extent).
        '''
        if 0 in tile.sel(band='analysis_extent').values:
            return False
        return True




    print('+++++++in tile_to_dir fn++++++++'))


    # Calculate the number of full tiles in x and y directions
    # num_x_tiles = datacube[0].x.size // tile_size
    # num_y_tiles = datacube[0].y.size // tile_size
    num_x_tiles = max(datacube[0].x.size + stride - 1, 0) // stride + 1
    num_y_tiles = max(datacube[0].y.size + stride - 1, 0) // stride + 1

    counter = 0
    counter_valid = 0
    counter_nomask = 0
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            # Calculate the start and end indices for x and y dimensions
            # for the current tile
            # x_start = x_idx * tile_size
            # y_start = y_idx * tile_size
            # x_end = min(stride * x_idx + tile_size, datacube[0].x.size)
            # y_end = min(stride * y_idx + tile_size, datacube[0].y.size)

            x_start = x_idx * stride
            y_start = y_idx * stride
            x_end = min(x_start + tile_size, datacube[0].x.size)
            y_end = min(y_start + tile_size, datacube[0].y.size)
            x_start = max(x_end - tile_size, 0)
            y_start = max(y_end - tile_size, 0)

            # Select the subset of data for the current tile
            parts = [part['band_data'][:, y_start:y_end, x_start:x_end] for part in datacube]

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

def main():
    datacube_name = 'datacube_vWebb.nc'
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    datacube = rxr.open_rasterio(Path(data_root,datacube_name ))
    save_path = Path(data_root, 'tiles')
    print(f"Writing tempfiles to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    '''
    tiles are normalised at this point.
    The tiles are saved in the format: tile_{event}_{date?}_num.tif, in event folders.
    TODO add further SEPERATE function to make trian test val split + txts
    
    '''

    # TODO iterate through datacube events
    # TODO grab the event name
    # TODO add the name to the saved tilename
    num_tiles, num_valid_tiles, num_nomask_tiles = tile_to_dir((datacube, save_path), tile_size=256, stride=256, event=event))
    tile_statistics[event] = [num_tiles, num_valid_tiles, num_nomask_tiles]
