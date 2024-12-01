from pathlib import Path
import rioxarray as rxr
import numpy as np
import xarray as xr
import rasterio 
from tqdm import tqdm



def main():
    print('>>>>>in main')
    base_path = Path(r"c:\users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TESTS\xrx")

    base_path.mkdir(exist_ok=True, parents=True)

    data = np.random.rand(2, 5, 5)
    layers = ['MASK', 'HH']
    coordinates = {'layer' : layers, 'y' :np.linspace(0,4,5), 'x' : np.linspace(0,4,5)}

    da1 = xr.DataArray(
                        data, 
                        dims =[ 'layer', 'y', 'x'],
                        coords = coordinates,  
                        attrs = {'country': 'UK', 'crs': 'EPSG:7777'}
    )   
    da1.attrs['landcover'] = 'forest'

    print('>>>>>da1 layers=', da1.coords['layer'].values)

    # da1.to_netcdf(base_path / 'test1.nc')

    # da1 = rxr.open_rasterio(base_path / 'test1.nc', decode_coords='all')
    # print('>>>>>da1 layers=', da1.coords['layer'].values)


    # return




    # da1.rio.write_crs("EPSG:7777", inplace=True)
    meta = {
        'driver': 'GTiff',
        'height': da1.sizes['y'],
        'width': da1.sizes['x'],
        'count': len(da1.layer),
        'dtype': da1.dtype,
        'crs': 'EPSG:7777',
        'transform': da1.rio.transform(),
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': 0
    }

    savepath = base_path / 'tile1.tif'
    with rasterio.open(savepath, 'w', **meta) as dst:
        for i , j in enumerate(layers):
            dst.write(data[i], i+1)
            dst.set_band_description(i+1, j)

    # da2= rxr.open_rasterio(savepath, decode_coords='all')
    with rasterio.open(savepath) as da2:

        print('>>>>>src tage=', da2.tags())

        print('>>>>>da2 description=', da2.descriptions)
    # print('>>>>>da2=', da2.coords['layers'].values)

    # print('>>>>>da2 crs=', da2.rio.crs)


if __name__ == '__main__':
    main()

