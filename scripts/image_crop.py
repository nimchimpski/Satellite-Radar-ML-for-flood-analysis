from osgeo import gdal
from pathlib import Path

# # Input and output configuration
# input_file = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\1data\2interim\TSX_test_final_image.tif" ) 
# output_file = input_file.parent / f"{input_file.stem}_cropped.tif"
# # Open the input GeoTIFF
# dataset = gdal.Open(input_file)
# if dataset is None:
#     raise FileNotFoundError(f"Could not open {input_file}")

# # Define the crop size in pixels
# crop_width = 1100
# crop_height = 1100

# # Get the image's size
# image_width = dataset.RasterXSize
# image_height = dataset.RasterYSize

# # Ensure the crop size does not exceed the image size
# if crop_width > image_width or crop_height > image_height:
#     raise ValueError("Crop dimensions exceed the size of the input image.")

# # Define the cropping window (centered)
# x_offset = (image_width - crop_width) // 2
# y_offset = (image_height - crop_height) // 2

# # Use gdal.Translate to crop the image
# gdal.Translate(
#     output_file,
#     dataset,
#     srcWin=[x_offset, y_offset, crop_width, crop_height]  # [x_offset, y_offset, width, height]
# )

# print(f"Cropped image saved as {output_file}")



# Input and output configuration
input_file = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\predictions_bkp\695972341_3_final_image.tif")
output_file = input_file.parent / f"{input_file.stem}_cropped.tif"

# Open the input GeoTIFF
dataset = gdal.Open(str(input_file))
if dataset is None:
    raise FileNotFoundError(f"Could not open {input_file}")

# Define crop size in pixels
crop_width = 1100
crop_height = 1100

# Get the original image size
image_width = dataset.RasterXSize
image_height = dataset.RasterYSize

# Ensure crop size does not exceed the image size
if crop_width > image_width or crop_height > image_height:
    raise ValueError("Crop dimensions exceed the size of the input image.")

# Define cropping window (centered)
x_offset = (image_width - crop_width) // 2
y_offset = (image_height - crop_height) // 2

# Use gdal.Translate to crop the image
gdal.Translate(
    str(output_file),
    dataset,
    srcWin=[x_offset, y_offset, crop_width, crop_height]  # [x_offset, y_offset, width, height]
)

# Open the cropped image to assign band descriptions
cropped_dataset = gdal.Open(str(output_file), gdal.GA_Update)

# Retain band descriptions (layer names)
for i in range(1, dataset.RasterCount + 1):
    layer_name = dataset.GetRasterBand(i).GetDescription()
    cropped_band = cropped_dataset.GetRasterBand(i)
    cropped_band.SetDescription(layer_name)

# Close datasets
cropped_dataset = None
dataset = None

print(f"Cropped image saved to {output_file} with layer names retained.")
