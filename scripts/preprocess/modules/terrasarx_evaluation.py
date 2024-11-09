import os
import rasterio
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from datetime import datetime
from pathlib import Path
import pyproj
from shapely.ops import transform


def terrasarx_evaluation(root_dir):
    # Initialize empty lists to store metadata
    sar_data = []
    mask_data = []

    # Directory paths
    sar_dir = root_dir / 'tsx_images'
    mask_dir = root_dir / 'tsx_masks'

    query = ['*IMAGE*', '*archive*']
    sar_bbox_list = []
    # mask_bbox_list = []
    # project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True).transform

    for q in query:
        print(f"---Querying: {q}")
        for file_path in root_dir.rglob(q):
            if file_path.is_file() and file_path.suffix in ['.tif', '.tiff']:
                with rasterio.open(file_path) as src:
                    # Check if CRS is already UTM; if not, reproject (optional)
                    print(f"---{file_path.name} has CRS {src.crs}")

                    if src.crs and src.crs.is_projected:

                        # Extract bounding box
                        bounds = box(*src.bounds)
                        timestamp = src.tags().get('TIMESTAMP')
                        date = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S') if timestamp else None
                        sar_bbox_list.append(bounds)
                        sar_data.append({
                            'file': file_path.name,
                            'bounds': bounds,
                            'date': date,
                            'type': 'SAR'
                        })
                        
                    else:
                        print(f"---ERROR not a UTM crs = {file_path.name}")

    sar_total_polygon = unary_union(sar_bbox_list)
    sar_total_area = sar_total_polygon.area / 10**6  # Convert to km^2
    return sar_data, mask_data, sar_total_area  

def main():
    root_dir = Path(r"Z:\1NEW_DATA\1data\2interim\dataset_DLR_TSX")
    sar_data, mask_data, sar_total_area = terrasarx_evaluation(root_dir)
    print('---len sar data= ',len(sar_data))
    print('---len mask data= ',len(mask_data))
    print(f'---sar total area= {sar_total_area} km^2')
    # print('---mask total area= ',mask_total_area)
    # all_data = pd.DataFrame(sar_data + mask_data)
    # print('---all data= ',all_data)

if __name__ == '__main__':
    main()

