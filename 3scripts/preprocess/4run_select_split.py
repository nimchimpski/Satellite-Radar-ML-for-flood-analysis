from pathlib import Path
import sys
import os
from tqdm import tqdm
from process_tiles_module import select_tiles_and_split
import click
import shutil
from helpers import get_incremental_filename, make_train_folders

@click.command()
@click.option("--testdata",is_flag=True)

def main(testdata):
    if testdata:
        click.echo("TEST DATA")
    else:
        click.echo("SERIOUS DATA")
    #######################!!!!!!!!!!!!!!!!!
    base_path = Path(r"Z:\1NEW_DATA\1data\2interim")
    main_dataset ="UNOSAT_FloodAI_Dataset_v2_norm"
    test_dataset = base_path / "tests" / "testdata_tiles_norm_med"
    train_ratio=0.6
    val_ratio=0.2
    test_ratio=0.2
    analysis_threshold=1
    mask_threshold=0.5
    MAKEFOLDER = True
    ########################################

    print('>>>mask threshold:', mask_threshold)
    print('>>>analysis threshold:', analysis_threshold)

    # testdata = True
    dataset = test_dataset
    new_dir = Path(r"Z:\1NEW_DATA\1data\3final\tests")
    # normalized_dataset = base_path / "tests" / "testdata_tiles_norm"
    # SERIOUS RUN
    if not testdata:
        dataset = base_path / main_dataset
        new_dir = Path(r"Z:\1NEW_DATA\1data\3final")

    dest_dir = get_incremental_filename(new_dir, f'{dataset.name}_split')
    print(f">>>new dir name: {dest_dir}")

    make_train_folders(dest_dir)

    total = 0
    rejected = 0
    tot_missing_extent = 0
    tot_missing_mask = 0
    low_selection = []
    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(dataset.rglob('normalized_minmax_tiles'))
    if not recursive_list:
        print(">>>No normalized folders found.")
        return
    
    # FILTER AND SPLIT
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        print(f">>> {folder.name}")
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
    print(f'>>>list of low selection folders: {low_selection}')   
    print(">>>tiles and texts match= ", (trainsize + valsize + testsize)== traintxtlen + valtxtlen + testtxtlen)
    print(f">>>Total selected tiles: {total}")
    print(f">>>Rejected tiles: {rejected}")
    print(f">>>Total missing extent: {tot_missing_extent}")
    print(f">>>Total missing mask: {tot_missing_mask}")

if __name__ == "__main__":
    main()