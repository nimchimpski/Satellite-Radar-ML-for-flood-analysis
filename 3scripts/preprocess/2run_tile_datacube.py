from pathlib import Path
import rioxarray as rxr
import numpy as np


def main():
    print('>>>>>in main')
    datacube_name = 'datacube_Vietnam_11alex.nc'
    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326\Vietnam_11")
    datacube = rxr.open_rasterio(Path(data_root,datacube_name ))
    print(f"################### tiling ###################")
    event = Path(data_root)

    # DO THE TILING AND GET THE STATISTICS
    total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask = tile_datacube(datacube, event, tile_size=256, stride=256)

    print(f'>>>>total num of tiles: {total_num_tiles}')
    print(f'>>>>num of saved tiles: {num_saved}')
    print(f'>>>num with NANs: {num_has_nans}')
    print(f'>>>num with no mask : {num_nomask}')
    print(f'>>>num with no valid layer or pixels : {num_novalid}')

    # check layers in tile
    tile_dir = Path(event, 'tiles') 
    # open a tile
    tile = rxr.open_rasterio(tile_dir / 'tile_Vietnam_11_0_0.tif')
    #print('>>>2 final saved tile= ', tile)
    print('---saved datacube crs = ', tile.rio.crs)

    
if __name__ == '__main__':
    main()
