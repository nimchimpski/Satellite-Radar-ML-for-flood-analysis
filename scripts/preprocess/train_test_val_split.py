import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios sum to 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Paths
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Create destination folders
    train_dir = dest_dir / "train"
    val_dir = dest_dir / "val"
    test_dir = dest_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    with open("train.txt", "w") as train,  open("val.txt", "w") as val,  open("test.txt", "w") as test:
        # Get a list of all files in the source directory
        files = list(source_dir.glob('*'))  # Modify '*' if you want a specific file extension
        random.shuffle(files)  # Shuffle files for random split

        # Calculate split indices
        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * val_ratio)

        # Split files into train, val, and test sets
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Copy files to their respective folders
        for file in train_files:
            shutil.copy(file, train_dir / file.name)
            # Write file paths to txt file
            train.write(f"{file}\n")
        for file in val_files:
            shutil.copy(file, val_dir / file.name)
            # Write file paths to txt file
            val.write(f"{file}\n")
        for file in test_files:
            shutil.copy(file, test_dir / file.name)
            # Write file paths to txt file
            test.write(f"{file}\n")

        print(f"---Total files: {len(files)}")
        print(f"---Train files: {len(train_files)}")
        print(f'---train.txt length: {len(train_files)}')
        print(f"---Validation files: {len(val_files)}")
        print(f'---val.txt length: {len(val_files)}')
        print(f"---Test files: {len(test_files)}")


