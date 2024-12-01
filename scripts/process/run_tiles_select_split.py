from pathlib import Path
import sys
import os
from tqdm import tqdm
from scripts.process_modules.process_tiles_module import select_tiles_and_split
import click
import shutil
import signal
from scripts.process_modules.process_tiles_module  import  make_train_folders, handle_interrupt, get_incremental_filename

@click.command()
@click.option("--testdata",is_flag=True)
def main(testdata):
    '''
    with MAKEFOLDER=False this can be used to asses how many tiles will be filtered out by the analysis and mask threshold before actually running the split.
    '''
    signal.signal(signal.SIGINT, handle_interrupt)

    if testdata:
        click.echo("TEST DATA")
    else:
        click.echo("SERIOUS DATA")
    #######################!!!!!!!!!!!!!!!!!
    MAKEFOLDER = True
    analysis_threshold=1
    mask_threshold=0.0

    src_base = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim")
    dst_base = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final")
    
    #TO SAVE TO C:/
    dest_dir = dst_base / 'train_input' 

    # TEST RUN
    if  testdata:
        dataset = src_base / "tests" / "TSX_normalized_tiles"
    # SERIOUS RUN
    else:
        dataset = src_base / 'TSX_normalized_tiles'
    
    dest_dir = get_incremental_filename(dest_dir, f'{dataset.name}_mt{mask_threshold}_split')
    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15

    ########################################
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f'>>>dest dir: {dest_dir}')   
    if not dest_dir.exists():
        print(f"Failed to create {dest_dir}")
        return
    print('>>>mask threshold:', mask_threshold)
    print('>>>analysis threshold:', analysis_threshold)
    make_train_folders(dest_dir)
    print('>>>dataset:', dataset)   

    total = 0
    rejected = 0
    tot_missing_extent = 0
    tot_missing_mask = 0
    low_selection = []
    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(dataset.rglob('*normalized_tiles'))
    print(f'>>>len recursive_list= {len(recursive_list)}')
    if not recursive_list:
        print(">>>No normalized folders found.")
        return
    
    # FILTER AND SPLIT
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        print(f">>> {folder.parent.name}/{folder.name}")
        foldertotal, selected_tiles, folderrejected, tot_missing_extent, tot_missing_mask = select_tiles_and_split(folder, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, MAKEFOLDER)
        if len(selected_tiles) < 10:
            low_selection.append(folder.parent.name)

        total += foldertotal
        rejected += folderrejected   
        tot_missing_mask += tot_missing_mask
        tot_missing_extent += tot_missing_extent 
 
        print(f">>>subtotal tiles: {total}")
        print(f">>>subtotal Rejected tiles: {rejected}")
        print(f">>>subtotal missing extent: {tot_missing_extent}")
        print(f">>>subtotal missing mask: {tot_missing_mask}")
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
    print(f'>>>total irrelevant files: {total - rejected - selected_tiles}')
    print(">>>tiles and texts match= ", (trainsize + valsize + testsize)== traintxtlen + valtxtlen + testtxtlen)
    
    
    print(f">>>Total missing extent: {tot_missing_extent}")
    print(f">>>Total missing mask: {tot_missing_mask}")
    print(f'>>>list of low selection folders: {low_selection}') 
    if MAKEFOLDER:
        print(f"Saved split data to {dest_dir}")
    else:
        print("NO TILES MADE - test run")

if __name__ == "__main__":
    main()