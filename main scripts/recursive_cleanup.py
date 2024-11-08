
import os
from pathlib import Path
import sys
module_path = Path(r"Z:\1NEW_DATA\3scripts\data-preprocessing\modules")
print('---module at ',module_path)
# Add this path to Python's search path (sys.path)
sys.path.append(str(module_path))
print('---sys path ',sys.path)
from .modules.renaming import remove_word_from_filename
from modules.organise_directory import delete_unnecessary_files   


if __name__ == "__main__":
    root_directory = Path(r"\\cerndata100\AI_Files\Users\AI_Flood_Service\1NEW_DATA\1data\2interim\TESTS\sample_s1s24326")
    for file_path in root_directory.rglob('*'):  # rglob('*') recursively lists all files and file_names
        delete_unnecessary_files(file_path)
    print('---deletions done_')

    for file_path in root_directory.rglob('*'):  # rglob('*') recursively lists all files and file_names

        remove_word_from_filename(file_path, 'epsg4326_')  # Remove the word 'epsg326' from the folder name