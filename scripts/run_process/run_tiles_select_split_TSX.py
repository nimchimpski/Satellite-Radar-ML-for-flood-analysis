from pathlib import Path
import sys
import os
from tqdm import tqdm
import click
import shutil
import signal
from scripts.process_modules.process_dataarrays_module  import  make_train_folders, get_incremental_filename, select_tiles_and_split
from scripts.process_modules.process_helpers import handle_interrupt

@click.command()
@click.option("--test",is_flag=True)
def main(test=None):
    '''
    with MAKEFOLDER=False this can be used to asses how many tiles will be filtered out by the analysis and mask threshold before actually running the split.
    '''
    signal.signal(signal.SIGINT, handle_interrupt)

    repo_dir = Path(__file__).resolve().parent.parent.parent
    print(f'>>>repo_dir= {repo_dir}')
    src_base = repo_dir / 'data' / '2interim' / 'TSX_tiles' / 'NORM_TILES_FOR_SELECT_AND_SPLIT_INPUT'
    dataset_name = None
    ############################################
    MAKEFOLDER = False
    analysis_threshold=1
    mask_threshold=0.3
    percent_under_thresh=0.25 # 0.001 = 1% 

    dst_base = repo_dir / 'data' / '4final' / 'train_INPUT'
    if test:
        click.echo("TEST DESTINATION")
        dst_base = repo_dir / 'data' / '4final' / 'test_INPUT'
    if not dst_base.exists():
        raise FileNotFoundError(f"Destination folder {dst_base} does not exist.")
    print(f"Destination folder: {dst_base}")
  

    train_ratio=0.829999
    val_ratio=0.17
    test_ratio=0.000001
    if test:
        train_ratio=0.001
        val_ratio=0.001
        test_ratio=0.998
    ########################################

    total = 0
    rejected = 0
    tot_missing_extent = 0
    tot_missing_mask = 0
    tot_under_thresh = 0
    low_selection = []

    # GET EVENT FOLDER NAME
    folders_to_process = list(f for f  in iter(src_base.iterdir()) if f.is_dir())
    # folder_to_process = folders_to_process[0]
    # print(f'>>>folder_to_process= {folder_to_process.name}')
    if len(folders_to_process) == 0:
        print(">>>No event folder found.")
        return
    elif len(folders_to_process) > 1:
        print(">>>Multiple event folders found.")
        return
    else:
        src_tiles = folders_to_process[0]
        print(f'>>>src_tiles_name= {src_tiles.name}')
    # parts = src_tiles.name.split('_')[:3]
    # print(f'>>>newname= {parts}')
    # newname = '_'.join(parts)
    # print(f'>>>newname= {newname}')
    dest_dir = get_incremental_filename(dst_base, f'{src_tiles.name}_mt{mask_threshold}_pcu{percent_under_thresh}')

    print(f'>>>source dir = {src_base}')
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f'>>>dest dir: {dest_dir}')   
    if not dest_dir.exists():
        print(f"Failed to create {dest_dir}")
        return
    print('>>>mask threshold:', mask_threshold)
    print('>>>analysis threshold:', analysis_threshold)
    make_train_folders(dest_dir)

    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(src_base.rglob('*normalized*'))
    print(f'>>>len recursive_list= {len(recursive_list)}')
    if not recursive_list:
        print(">>>No normalized folders found.")
        return
    
    # FILTER AND SPLIT
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        # GET NUMBER OF FILES IN FOLDER

        print(f"\n>>>>>>>>>>>>>>>>>>> AT FOLDER {folder.name}>>>>>>>>>>>>>>>")
        foldertotal, folder_selected, folderrejected, folder_missing_extent, folder_missing_mask, folder_under_thresh = select_tiles_and_split(folder, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, percent_under_thresh, MAKEFOLDER)
        print(f'>>>folder total= {foldertotal}')
        print(f'>>>folder selected= {folder_selected}')
        print(f'>>>folder rejected= {folderrejected}')
        print(f'>>>folder under threshold= {folder_under_thresh}')
        if folder_selected < 10:
            low_selection.append(folder.name)

        total += foldertotal
        rejected += folderrejected   
        tot_missing_mask += folder_missing_mask
        tot_missing_extent += folder_missing_extent 
        tot_under_thresh += folder_under_thresh    
 
        print(f">>>subtotal tiles: {total}")
        print(f">>>subtotal Rejected tiles: {rejected}")
        # print(f">>>subtotal missing extent: {tot_missing_extent}")
        # print(f">>>subtotal missing mask: {tot_missing_mask}")
        print(f">>>subtotal under threshold: {tot_under_thresh}")
        
    print('>>>>>>>>>>>>>>>> DONE >>>>>>>>>>>>>>>>>')
    with open(dest_dir / "train.txt", "r") as traintxt,  open(dest_dir / "val.txt", "r") as valtxt,  open(dest_dir / "test.txt", "r") as testtxt:
        traintxtlen = sum(1 for _ in traintxt)
        print('>>>len traint.txt= ', traintxtlen)
        valtxtlen = sum(1 for _ in valtxt)
        print('>>>len val.txt= ', valtxtlen)
        testtxtlen = sum(1 for _ in testtxt)
        print('>>>len test.txt= ', testtxtlen) 

    # print(">>>total txt len = ", traintxtlen + valtxtlen + testtxtlen)
    trainsize = len(list((dest_dir / 'train').iterdir()))
    valsize = len(list((dest_dir / 'val').iterdir()))
    testsize = len(list((dest_dir / 'test').iterdir()))
    selected_tiles = trainsize + valsize + testsize
    print(f">>>Total all original tiles: {total}")
    print(f'>>>total t+v+t tiles: {selected_tiles}')
    print(f">>>Rejected tiles: {rejected}")
    print(f'>>>total under threshold included: {tot_under_thresh}')
    print(f'>>>total irrelevant files: {total - rejected - selected_tiles}')
    print(">>>tiles and texts match= ", (trainsize + valsize + testsize)== traintxtlen + valtxtlen + testtxtlen)
    
    
    # print(f">>>Total missing extent: {tot_missing_extent}")
    # print(f">>>Total missing mask: {tot_missing_mask}")
    print(f'>>>list of low selection folders: {low_selection}') 
    if MAKEFOLDER:
        print(f"Saved split data to {dest_dir}")
    else:
        print("NO TILES MADE - test run")

if __name__ == "__main__":
    main()