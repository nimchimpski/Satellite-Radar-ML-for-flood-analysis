from pathlib import Path
import shutil
from tqdm import tqdm

def combine_txt_files(txt_file1, txt_file2, output_file):
    """
    Combine two .txt files into one, removing duplicates if necessary.
    """
    # Read entries from both files
    with open(txt_file1, "r") as f1, open(txt_file2, "r") as f2:
        entries1 = f1.readlines()
        entries2 = f2.readlines()
    
    # Combine and remove duplicates
    combined_entries = list(set(entries1 + entries2))
    
    # Save to the new file
    with open(output_file, "w") as out:
        out.writelines(sorted(combined_entries))  # Sort for consistency

def combine_datasets(dataset1_path, dataset2_path, output_path):
    """
    Combine train/val/test splits and their corresponding .txt files,
    ensuring the original files are preserved.
    """
    splits = ["train", "val", "test"]
    
    for split in splits:
        # Paths to the .txt files
        txt_file1 = Path(dataset1_path) / f"{split}.txt"
        txt_file2 = Path(dataset2_path) / f"{split}.txt"
        output_txt = Path(output_path) / f"{split}.txt"
        
        # Combine .txt files
        combine_txt_files(txt_file1, txt_file2, output_txt)
        print(f"Combined {split}.txt saved to {output_txt}")
        
        # Combine image/tiles directories
        split_dir1 = Path(dataset1_path) / split
        split_dir2 = Path(dataset2_path) / split
        output_split_dir = Path(output_path) / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files from both datasets
        for file in tqdm(split_dir1.glob("*"), desc=f"Copying {split} files"):
            dest_file = output_split_dir / file.name
            if not dest_file.exists():
                shutil.copy(file, dest_file)  # Copy file to the destination
            else:
                # Handle duplicates by renaming
                new_name = f"{file.stem}_copy{file.suffix}"
                shutil.copy(file, output_split_dir / new_name)
        
        for file in tqdm(split_dir2.glob("*"), desc=f"Copying {split} files"):
            dest_file = output_split_dir / file.name
            if not dest_file.exists():
                shutil.copy(file, dest_file)  # Copy file to the destination
            else:
                # Handle duplicates by renaming
                new_name = f"{file.stem}_copy{file.suffix}"
                shutil.copy(file, output_split_dir / new_name)
        
        print(f"Combined {split} directory saved to {output_split_dir}")



def main():
    ##################################
    combine_folder = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\to_combine_###")
    output_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\3final\train_input_###")
    if not (combine_folder.exists()) or not (combine_folder.is_dir()):
        print(f" folder {combine_folder} does not exist.")
        return
    ##################################

    # get folders in the combine folder
    combine_folders = [f for f in combine_folder.iterdir() if f.is_dir()]
    print(f"Combine folders: {combine_folders}")

    combine_datasets(
    dataset1_path = combine_folders[0] ,
    dataset2_path = combine_folders[1],
    output_path = output_path
    )

if __name__ == "__main__":
    main()