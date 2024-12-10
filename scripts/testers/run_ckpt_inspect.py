import torch
from pathlib import Path
from collections import OrderedDict
from scripts.train_modules.train_classes import UnetModel


# Load the checkpoint
ckpt_path = Path(r"C:\Users\floodai\UNOSAT_FloodAI_v2\4results\checkpoints\TSX_logclipmm_g_mt0.3__BS16__EP10_WEIGHTED_BCE_copy.ckpt")  # Replace with your actual file path
checkpoint = torch.load(ckpt_path, map_location="")


def clean_checkpoint_keys(state_dict):
    """Fix the keys in the checkpoint by removing extra prefixes."""
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.")
        elif key.startswith("model."):
            new_key = key.replace("model.", "")
        else:
            new_key = key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict

cleaned_state_dict = clean_checkpoint_keys(checkpoint["state_dict"])

model = UnetModel(
    encoder_name="resnet34",  # Replace with the encoder used during training
    in_channels=1,
    classes=1,
    pretrained=False  # Set to False if you're loading a custom-trained model
)

model.load_state_dict(cleaned_state_dict)
model.eval()

print(f'---ckpt type= {type(checkpoint)}')
# Inspect the keys in the checkpoint
print(f'---ckpt keys= {checkpoint.keys()}')

# Print checkpoint keys
print("cleaned Checkpoint keys:")
print(list(cleaned_state_dict.keys()))

# Print model's expected keys
print("Model keys:")
print(list(model.state_dict().keys()))
