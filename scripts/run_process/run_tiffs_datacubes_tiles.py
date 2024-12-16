
from tqdm import tqdm
from pathlib import Path
from scripts.process_modules.process_tiffs_module import create_event_datacubes
from scripts.process_modules.process_dataarrays_module import tile_datacube_rasterio
import rioxarray as rxr
import time
from scripts.process_modules.process_tiffs_module import create_slope_from_dem
import shutil



# cd Y:/1NEW_DATA/1data/2interim/dset_DLR_S1S2_bycountry_4326

start = time.time()



    # MAKE DATACUBES
    create_event_datacubes(source, save_cube_path, VERSION="v1")

    # print(">>>datacubes saved in: ", save_cube_path)

    # save_tiles_path = data_root / "dset_DLR_S1S2_bycountry_4326_tiles"
    # if save_tiles_path.exists():
    #     # delete the folder and create a new one
    #     save_tiles_path.unlink()
    #     return
    # save_tiles_path.mkdir(exist_ok=True, parents=True)

        
    # total_num_tiles = 0
    # total_saved = 0
    # total_has_nans = 0
    # total_novalid_layer = 0
    # total_novalid_pixels = 0
    # total_nomask = 0
    # total_nomask_pixels = 0

    # cubes = list(save_cube_path.rglob("*.nc"))   
    # for cube in tqdm(cubes, desc="### Datacubes tiled"):
    #     print("cube=", cube.name)
    #     print(f"################### tiling ###################")

    #     # DO THE TILING AND GET THE STATISTICS

    #     num_tiles, num_saved, num_has_nans, num_novalid_layer, num_novalid_pixels, num_nomask, num_nomask_pixels = tile_datacube_rasterio(cube, save_tiles_path, tile_size=256, stride=256)

    #     total_num_tiles += num_tiles
    #     total_saved += num_saved
    #     total_has_nans += num_has_nans
    #     total_novalid_layer += num_novalid_layer
    #     total_novalid_pixels += num_novalid_pixels
    #     total_nomask += num_nomask
    #     total_nomask_pixels += num_nomask_pixels

    # print(f'>>>>total num of tiles: {total_num_tiles}')
    # print(f'>>>>num of saved tiles: {num_saved}')
    # print(f'>>>num with NANs: {num_has_nans}')
    # print(f'>>>num with no valid layer : {num_novalid_layer}')
    # print(f'>>>num with no valid  pixels : {num_novalid_pixels}')
    # print(f'>>>num with no mask : {num_nomask}')
    # print(f'>>>num with no mask pixels : {total_nomask_pixels}')
    # print(f'>>>all tiles tally: {total_num_tiles == total_saved + total_has_nans + total_novalid_layer + total_novalid_pixels + total_nomask + total_nomask_pixels}')
        # check layers in tile
        # tile_dir = Path(event, 'tiles') 
        # # open a tile
        # tile = rxr.open_rasterio(tile_dir / 'tile_Vietnam_11_0_0.tif')
        # #print('>>>2 final saved tile= ', tile)

    end = time.time()
    # time taken in minutes to 2 decimal places
    print(f"Time taken: {((end - start) / 60):.2f} minutes")


if __name__ == "__main__":
    main()