
# MODULES
from modules.check_int16_exceedance import check_int16_exceedance
from modules.helpers import *
from tile_datacube import tile_datacube 
from modules.organise_directory import collect_images
from modules.preprocess_tiffs import create_event_datacubes, process_terraSARx_data
from modules.reproject_tifs_to_4326 import reproject_tifs_to_4326
'''


TODO check process in place to organise the dataroot / input files such as they are correct for this script.
TODO THERE IS A PERSISTANT HIDDEN PROBLEM WITH WRONG VALUES AFTER CASTING.
TODO add rtc function 
'''

def main():

    data_root = Path(r"Z:\1NEW_DATA\1data\2interim\TESTS\terrasarx_test_data\current_images")
    # CREATE A FOLDER AT THE TOP OF EACH EVENT DIR WITH NECESSARY TIFS
    process_terraSARx_data(data_root)

    #MATCH ALL CRS TO THE SAR IMAGE
    reproject_tifs_to_4326(data_root)


    #create_event_datacubes(data_root)

    # for event in data_root.iterdir():
    #    if event.is_dir() and any(event.iterdir()): # select event folders
    #        print(f"############## {event.name}   TILING ########################: ")
    #        for file in tqdm(event.iterdir(), desc=':::::iterate files in event folders'):
    #            if file.suffix == '.nc':
    #                #datacube = xr.open_dataset(file)
    #                #print('---datacube= ',datacube)
    #                print(f">>>>>>>>>>>>>>>>>>>>>>> TILING {event.name}<<<<<<<<<<<<<<<<: ")
    #                total_num_tiles, num_saved, num_has_nans, num_novalid, num_nomask = tile_datacube(file, event, tile_size=256, stride=256)

    # print(f'>>>>total num of tiles: {total_num_tiles}')
    # print(f'>>>>num of saved tiles: {num_saved}')
    # print(f'>>>num witno valid data/analysis extent layer: {num_has_nans}')
    # print(f'>>>num with no mask : {num_nomask}')
    # print(f'>>>num with no valid layer : {num_novalid}')

if __name__ == "__main__":
    main()
