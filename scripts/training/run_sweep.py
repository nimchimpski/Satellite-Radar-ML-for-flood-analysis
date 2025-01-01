# Import the W&B Python Library and log into W&B
import wandb
from pathlib import Path

# 1: Define objective/training function
def objective(config):
    try:
        score = config.focal_alpha + config.focal_gamma
    except AttributeError as e:
        print("---Current config:", config)

        print(f"---Error accessing config attributes: {e}")
        raise
    return score

wandb.init()
print("---Current wandb.config:", wandb.config)  # Debug print
score = objective(wandb.config)
wandb.log({"score": score})
sweep_configuration = {
    "name": "focal_loss_sweep",
    "project": "TSX",
    "program": 'train_new.py',
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "focal_alpha": {"value": 6.5},
        "focal_gamma": {"values": [0.40,  0.7, ]},
    },
}

    # 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="TSX")
print(f"---Sweep ID: {sweep_id}")  # Debug print
    
train = Path("train_new.py")
wandb.agent(sweep_id, function=train, count=10)
