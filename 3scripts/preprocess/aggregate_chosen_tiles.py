import os
import shutil
from pathlib import Path
from tqdm import tqdm

def find_data_folders(base_path):
    """Recursively search  to find 'train', 'val', and 'test' folders.
    create lists of paths to each folder type."""
    print('+++in find_data_folders')
    data_folders = {'train': [], 'val': [], 'test': []}
    for root, dirs, _ in os.walk(base_path):
        # print('---root = ', root)
        for d in dirs:
            if d in data_folders:
                data_folders[d].append(Path(root) / d)
    # print('---data_folders = ', data_folders)
    return data_folders



def copy_data_and_generate_txt(data_folders, destination):
    """Copy all files into a centralized destination and create .txt files for train, val, and test."""
    print('+++in copy_data_and_generate_txt')
    dest_paths = {key: destination / key for key in data_folders}
    txt_files = {key: destination / f"{key}.txt" for key in data_folders}
    
    # Ensure destination folders exist
    for path in dest_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Process each folder type (train, val, test)
    for key, folders in data_folders.items():
        with open(txt_files[key], 'w') as txt_file:
            for folder in folders:
                file_list = list(folder.glob('*'))  # Get all files in the folder
                for file_path in tqdm(file_list, desc=f"Copying {key} files from {folder}", unit="file"):
                    if file_path.is_file():  # Copy only files
                        dest_file_path = dest_paths[key] / file_path.name
                        shutil.copy(file_path, dest_file_path)
                        # Write the path of the copied file to the .txt file
                        txt_file.write(str(dest_file_path) + '\n')
        print(f"Copied {key} files and generated {key}.txt")

# Define the source and destination paths
base_path = Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\1data\2interim\UNOSAT_FloodAI_Dataset_v2_norm")  # Replace with the top-level path of your source data
# base_path = Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\1data\2interim\TESTS\tile_norm_testdata_norm")  # Replace with the top-level path of your source data
destination = Path(r"\\cerndata100\AI_Files\Users\AI_flood_Service\1NEW_DATA\1data\3final\ds_v2_selection")   # Replace with your desired destination path

# check base path exists
if not base_path.exists():
    raise FileNotFoundError(f"Source path '{base_path}' does not exist.")
# Find all train, val, test folders within the source path
data_folders = find_data_folders(base_path)

# Copy data and create accompanying .txt files
copy_data_and_generate_txt(data_folders, destination)

print("Data aggregation complete!")
