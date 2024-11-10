import wandb
from pathlib import Path
import torch
import torch.nn as nn



def initialize_wandb(project, job_type, run_name):
    """
    Initializes WandB if not already initialized.
    
    Args:
    - project (str): The name of the WandB project.
    - job_type (str): The type of job (e.g., 'train', 'reproduce').
    - run_name (str): The name of the WandB run.
    
    Returns:
    - wandb.run: The active WandB run.
    """
    # Check if WandB is already initialized
    if wandb.run is None:
        # Initialize WandB
        run = wandb.init(
            project=project,
            job_type=job_type,
            name=run_name
        )
        return run
    else:
        # If already initialized, return the existing run
        return wandb.run
    



class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, classes, kernel_size=3, padding=1)  # Output with 2 channels

        # Optional: Use an upsampling layer to match the output size with the ground truth
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        
        # Optional: Use upsampling if necessary to match the size
        x = self.upsample(x)
        return x


