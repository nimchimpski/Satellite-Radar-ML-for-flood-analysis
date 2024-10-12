
import os
from pathlib import Path
import sys
module_path = Path(r"Z:\1NEW_DATA\3scripts\data-preprocessing\modules")
print('---module at ',module_path)
# Add this path to Python's search path (sys.path)
sys.path.append(str(module_path))
print('---sys path ',sys.path)
from renaming_module import remove_word_from_filename

def delete_unnecessary_files(file_name):
    """
    This function iterates through all subfile_names in the root directory
    and deletes any .xml files and files that start with 'sentinel12'.
    Args:
    - root_dir (str or Path): The root directory to start the search.
    """
    # Iterate through all subdirectories and files in the root directory
    if file_name.is_file():
        # Delete .xml files
        if file_name.suffix == '.xml':
            print(f"Deleting {file_name}")
            file_name.unlink()  # Remove the file
            
        # Delete files that start with 'sentinel12'
        elif file_name.suffix != '.json':
            if file_name.name.startswith('sentinel12'):
                print(f"Deleting {file_name}")
                file_name.unlink()  # Remove the file
            elif 'sentinel12_s2' in file_name.name:
                print(f"Deleting {file_name}")
                file_name.unlink()  # Remove the file

if __name__ == "__main__":
    root_directory = Path(r"\\cerndata100\AI_Files\Users\AI_Flood_Service\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    for file_path in root_directory.rglob('*'):  # rglob('*') recursively lists all files and file_names
        delete_unnecessary_files(file_path)
    print('---deletions done_')

    for file_path in root_directory.rglob('*'):  # rglob('*') recursively lists all files and file_names

        remove_word_from_filename(file_path, 'epsg4326_')  # Remove the word 'epsg326' from the folder name