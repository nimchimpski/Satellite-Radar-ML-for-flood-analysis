
from pathlib import Path
import shutil

dataset = Path(r"Y:\1NEW_DATA\1data\2interim\ALL TSX")
save_path= Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_all_processing\TSX_kmls")
for folder in dataset.iterdir():
    if folder.is_dir():
        for event in folder.iterdir(): 
            code = '_'.join(event.name.split('_')[4:])
            # print(event.name)
            print(code)
            kml = next(event.rglob('STD_MRES_HH*.kml'))
            shutil.copy(kml, save_path )

# get num files in save_path
num_files = len(list(save_path.rglob('*.kml')))