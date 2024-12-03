

import torch
import numpy as np
import xarray as xr
from pathlib import Path
import tifffile as tiff
from model_definition import UnetModel  # Adjust the import to your model definition
from scripts.process_modules.process_tiles_module import log_clip_minmaxnorm  # Adjust the import to your preprocessing function

checkpoint_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\4results\checkpoints")
checkpoint = xxxxx

# 1. Load the model
def load_model(checkpoint_path, device='cuda'):
    model = UnetModel(encoder_name='resnet34', in_channels=2, classes=1)  # Update based on your architecture
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 2. Preprocess input data
def preprocess_tile(tile, device):
    """
    Preprocess SAR tile for inference.
    - Normalize or log-transform the data to match training preprocessing.
    - Convert to PyTorch tensor and move to the correct device.
    """
    # Log-transform and normalize (example preprocessing; adjust based on your pipeline)
    log_clip_minmaxnorm(tile)

    # Convert to PyTorch tensor
    tensor = torch.tensor(tile_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

# 3. Post-process predictions
def postprocess_predictions(predictions, threshold=0.5):
    """
    Post-process model predictions.
    - Apply a threshold to convert logits to binary mask.
    """
    binary_mask = (predictions > threshold).float().squeeze(0).cpu().numpy()
    return binary_mask

# 4. Run inference
def run_inference(model, tile, device='cuda'):
    preprocessed_tile = preprocess_tile(tile, device)
    with torch.no_grad():
        predictions = model(preprocessed_tile)
    return postprocess_predictions(predictions)

# 5. Main function
def main(input_path, checkpoint_path, output_path, device='cuda'):
    # Load the trained model
    model = load_model(checkpoint_path, device)

    # Load the input SAR tile
    input_tile = xr.open_dataset(input_path)['tile']  # Adjust based on your dataset format

    # Run inference
    predictions = run_inference(model, input_tile, device)

    # Save predictions
    output_file = Path(output_path) / f"{Path(input_path).stem}_prediction.tif"
    tiff.imwrite(output_file, predictions.astype(np.uint8) * 255)  # Save as binary mask
    print(f"Predictions saved to {output_file}")

# Run the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with a trained U-Net model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input SAR tile (GeoTIFF or NetCDF).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the output prediction.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on ('cuda' or 'cpu').")
    args = parser.parse_args()

    main(args.input, args.checkpoint, args.output, args.device)
