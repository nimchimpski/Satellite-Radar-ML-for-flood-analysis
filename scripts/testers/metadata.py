
import json
from pathlib import Path
input_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\4final\predict_INPUT\IMAGE_HH_tiles\tile_metadata.json")

def read_minmax_from_json(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    return data
metadata = read_minmax_from_json(input_path)

for tile_info in metadata:
    print(tile_info["tile_name"], tile_info["x_start"], tile_info["y_start"])
