

from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm
import shutil  # For safely replacing files


def main():
    tiles_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_process1\archive\dims_op_oc_dfd2_695959729_1\695959729_1_MASK.tif")


    with rasterio.open(tiles_path) as src:
        data = src.read()
        print(f">>> Original mask stats: min={data.min()}, max={data.max()}, unique={np.unique(data)}")



if __name__ == "__main__":
    main()
