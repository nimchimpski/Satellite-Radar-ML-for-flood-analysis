from pathlib import Path
import shutil
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scripts.preprocess_modules.process_tiffs_module import match_dem_to_sar
from scripts.preprocess_modules.process_tiffs_module import calculate_and_normalize_slope, create_event_datacubes_tsx

data_root = Path(r'C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_process1\archive')


if False:
# if True:
    # MOVE THE MASK, IMAGE, AND DEM TO THE EXTRACTED FOLDER
    for event in data_root.iterdir():
        print(f'---event={event.name}')
        extract_folder = event / 'extracted'
        extract_folder.mkdir(exist_ok=True)

        # COPY THE MASK
        mask = list(event.rglob('*MASK.tif'))[0]
        print(f'---mask={mask.name}')
        if not (extract_folder / mask.name).exists():
            shutil.copy(mask, extract_folder)

        mask_code = mask.name.split('_')[0]
        print(f'---mask_code= ',mask_code)

        # COPY THE IMAGES
        image = list(event.rglob('*IMAGE*.tif'))[0]
        print(f'---imagename ={image.name}')
        image_rename = extract_folder / f'{mask_code}_image.tif'
        print(f'---image_rename={image_rename.name}')
        if not image_rename.exists():
            shutil.copy(image, image_rename)

        # COPY THE DEM
        dem = list(event.rglob('*DEM*.tif'))[0]
        print(f'---dem={dem.name}')
        if not (extract_folder / dem.name).exists():
            shutil.copy(dem, extract_folder)

        matched_dem = extract_folder / f'{mask_code}_dem.tif'

        match_dem_to_sar(dem, image, matched_dem)
        # CHECK THE NEW DEM
        with rasterio.open(image) as image_src:
            image_bounds = image_src.bounds
            target_crs = image_src.crs
            width = image_src.width
            height = image_src.height
            transform = image_src.transform
        with rasterio.open(matched_dem) as src:
            with rasterio.open(image) as src:
                print(f'---bounds match={src.bounds == image_bounds}')
                print(f'---crs match={src.crs == target_crs}')
                print(f'---width match={src.width == width}')
                print(f'---height match={src.height == height}')
                print(f'---transform match={src.transform == transform}')
                print(f'---count match={src.count == src.count}')

        calculate_and_normalize_slope(matched_dem, mask_code)

        with rasterio.open(extract_folder / f'{mask_code}_slope_norm.tif') as src:
            print(f'---slope min={src.read().min()}')
            print(f'---slope max={src.read().max()}')

create_event_datacubes_tsx(data_root)

