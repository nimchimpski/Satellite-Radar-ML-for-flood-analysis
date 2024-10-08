import re
from pathlib import Path

base_path_root = r"X:\1NEW_DATA\1data\2interim"
base_path = Path(base_path_root) / "dataset_DLR_S1S2_bycountry"

def rename(base_path):
    '''
    rename folders by removing everything apart from digits between 1 and 99
    '''
    if not base_path.exists():
        print(f"---Path does not exist: {base_path}")
        return
    
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            print(f"---Processing folder: {folder_path.name}")
            
            # Use regex to find digits between 1 and 99
            digits = re.findall(r'\d{1,2}', folder_path.name)  # Matches 1 or 2 digit numbers
            
            # Filter out any numbers greater than 99 (in case there are such numbers)
            digits = [digit for digit in digits if 1 <= int(digit) <= 99]
            
            if digits:
                # Join the matched digits into a new folder name
                new_folder_name = '_'.join(digits)
                new_folder_path = folder_path.parent / new_folder_name
                
                # Check if the new folder already exists to avoid errors
                if not new_folder_path.exists():
                    # Rename the folder
                    print(f"Renaming: {folder_path} -> {new_folder_path}")
                    folder_path.rename(new_folder_path)
                else:
                    print(f"Folder with the name {new_folder_path} already exists. Skipping.")
            else:
                print(f"No digits between 1 and 99 found in {folder_path.name}. Skipping.")

# Call the function
rename(base_path)
