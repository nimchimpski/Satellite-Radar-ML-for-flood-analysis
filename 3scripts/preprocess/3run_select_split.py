from pathlib import Path
import sys
import os
from tqdm import tqdm
from process_tiles_module import select_tiles_and_split
import click
import shutil
from helpers import get_incremental_filename



# Add the parent directory of the current file to the system path
# Get the root directory and add it to sys.path
# Add the parent directory of the current file to sys.path


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
    train_ratio=0.7 
    val_ratio=0.15
    test_ratio=0.15
    analysis_threshold=1
    mask_threshold=0.5
    MAKEFOLDER = False

    print('>>>mask threshold:', mask_threshold)
    print('>>>analysis threshold:', analysis_threshold)

    # testdata = True
    dataset = test_dataset
    # normalized_dataset = base_path / "tests" / "testdata_tiles_norm"
    # SERIOUS RUN
    if not testdata:
        dataset = base_path / main_dataset
    
    new_dir = Path(r"Z:\1NEW_DATA\1data\3final" , f'{dataset.name}_split')
    
    dest_dir = get_incremental_filename(new_dir, f'{dataset.name}_split')
    print(f">>>new dir name: {dest_dir}")


    # Create destination folders
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    rejected = 0
    tot_missing_extent = 0
    tot_missing_mask = 0
    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(dataset.rglob('normalized_minmax_tiles'))
    if not recursive_list:
        print(">>>No normalized folders found.")
        return
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        print(f">>> {folder.name}")
        # FILTER AND SPLIT
        foldertotal, folderrejected, tot_missing_extent, tot_missing_mask = select_tiles_and_split(folder, dest_dir, train_ratio, val_ratio, test_ratio, analysis_threshold, mask_threshold, MAKEFOLDER)
        total += foldertotal
        rejected += folderrejected   
        tot_missing_mask += tot_missing_mask
        tot_missing_extent += tot_missing_extent 

    print(f">>>Total tiles: {total}")
    print(f">>>Rejected tiles: {rejected}")
    print(f">>>Total missing extent: {tot_missing_extent}")
    print(f">>>Total missing mask: {tot_missing_mask}")

if __name__ == "__main__":
    main()