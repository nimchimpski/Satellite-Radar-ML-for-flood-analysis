
from tqdm import tqdm
from pathlib import Path
from ..modules.tiff_folders_to_datacubes import create_event_datacubes
from ..modules.tile_datacube import tile_datacube
import rioxarray as rxr



# cd Y:/1NEW_DATA/1data/2interim/dset_DLR_S1S2_bycountry_4326



def main():
    data_root = Path(r"\\cerndata100\AI_Files\Users\AI_Flood_Service\1NEW_DATA\1data\2interim\tests")
    data_name = "dset_DLR_S1S2_bycountry_4326_test"
    source = data_root / data_name
    save_cube_path = data_root / "dset_DLR_S1S2_bycountry_4326_datacubes"

    # create_event_datacubes(source, save_cube_path, VERSION="v1")

    print(">>>datacubes saved in: ", save_cube_path)


    cubes = list(save_cube_path.rglob("*.nc"))   
    for cube in tqdm(cubes, desc="### Datacubes tiled"):
        print("cube=", cube.name)
        print(f"################### tiling ###################")
        save_tiles_path = data_root / "dset_DLR_S1S2_bycountry_4326_tiles"

        ##DO THE TILING AND GET THE STATISTICS
        total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask = tile_datacube(cube, save_tiles_path, tile_size=256, stride=256)

        print(f'>>>>total num of tiles: {total_num_tiles}')
        print(f'>>>>num of saved tiles: {num_saved}')
        print(f'>>>num with NANs: {num_has_nans}')
        print(f'>>>num with no mask : {num_nomask}')
        print(f'>>>num with no valid layer or pixels : {num_novalid}')

        # # check layers in tile
        # tile_dir = Path(event, 'tiles') 
        # # open a tile
        # tile = rxr.open_rasterio(tile_dir / 'tile_Vietnam_11_0_0.tif')
        # #print('>>>2 final saved tile= ', tile)

    


if __name__ == "__main__":
    main()