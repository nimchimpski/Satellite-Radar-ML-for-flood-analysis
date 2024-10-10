import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
from pathlib import Path
import json
import time
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling




# Function to get the central coordinates from a TIFF file
def get_central_coordinate_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        # Get the bounding box of the dataset
        bbox = dataset.bounds
        if not bbox:
            print("---Bounding box not found in the TIFF file")
            return None
        # Calculate the central coordinate in the native CRS
        central_lon = (bbox.left + bbox.right) / 2
        central_lat = (bbox.bottom + bbox.top) / 2

        return central_lat, central_lon
    
def get_crs_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        if not dataset.crs:
            print("---CRS not found in the TIFF file")
            crs = None
        else:
            print(f"---CRS: {dataset.crs}")
            crs = dataset.crs
        return crs 
    
# Function to get the country name using reverse geocoding
def reverse_geolocate_country_utmzone(lat, lon):
    geolocator = Nominatim(user_agent="floodai")
    try:
        print(f"---Reverse geocoding lat: {lat}, lon: {lon}")  # Debug: Print coordinates being geocoded
        time.sleep(1)  # Add a delay between requests
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'country' in location.raw['address']:
            country = location.raw['address']['country']
        else:
            country = "Unknown"
    
        # Calculate UTM zone from longitude
        utm_zone = int((lon + 180) / 6) + 1

        # Determine EPSG code based on hemisphere (north/south)
        if lat >= 0:
            epsg_code = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
        else:
            epsg_code = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere

        utmzone =  epsg_code
        return country, utmzone
    except Exception as e:
        print(f"---Error during reverse geocoding: {e}")
        return "Unknown"
    
def reproject_to_epsg4326(tile_path, utm_crs):
    dst_crs = CRS.from_epsg(4326)  # WGS84 EPSG:4326

    with rasterio.open(tile_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Save the reprojected raster to a new file or overwrite the same file
        output_file = tile_path.with_name(f"reprojected_{tile_path.name}")
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    print(f"---Reprojected tile saved as: {output_file}")