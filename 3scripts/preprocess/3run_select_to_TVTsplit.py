from pathlib import Path
import sys
import os
from tqdm import tqdm
from process_tiles_module import select_tiles_and_split
import click

# Add the parent directory of the current file to the system path
# Get the root directory and add it to sys.path
root_dir = Path(__file__).resolve().parents[2]  # Adjust level as needed to reach the root
sys.path.insert(0, str(root_dir))
base_path = Path(r"Z:\1NEW_DATA\1data\2interim")
@click.command()
@click.option("--test",is_flag=True)

def main(test):

    if test:
        click.echo("TEST RUN")
    else:
        click.echo("SERIOUS RUN")

    base_path = Path(r"Z:\1NEW_DATA\1data\2interim")

    if test:
        normalized_dataset = base_path / "tests" / "testdata_tiles_norm"
    # normalized_dataset = base_path / "tests" / "testdata_tiles_norm"
    # SERIOUS RUN
    else:
        normalized_dataset = base_path / "UNOSAT_FloodAI_Dataset_v2_norm"

    dest_dir = Path(r"Z:\1NEW_DATA\1data\4final" , f'{normalized_dataset.name}_TVTsplit')

    # Create destination folders
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    rejected = 0
    #GET ALL NORMALIZED FOLDERS
    recursive_list = list(normalized_dataset.rglob('normalized_minmax_tiles'))
    for folder in  tqdm(recursive_list, desc="TOTAL FOLDERS"):
        print(f"---: {folder.name}")
        # FILTER AND SPLIT
        foldertotal, folderrejected = select_tiles_and_split(folder, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        total += foldertotal
        rejected += folderrejected    

    print(f"---Total tiles: {total}")
    print(f"---Rejected tiles: {rejected}")

if __name__ == "__main__":
    main()