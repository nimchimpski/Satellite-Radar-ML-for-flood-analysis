# Import the W&B Python Library and log into W&B
import wandb

wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.focal_alpha + config.focal_gamma
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "program": 'train_new.py',
    "name": "sweep1",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "batch_size": {"value": 16},
        "epochs": {"value": 15},
        # "focal_alpha": {"values": [ 0.2, 0.4, 0.6, 0.8]},
        # "focal_gamma": {"values": [0.1,  0.3,  0.5,  0.7 ]},
        "subset_fraction": {"value": 1},
        "focal_alpha": {"value": 6.5},
        "focal_gamma": {"values": [0.15,  0.25, ]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="TSX")

wandb.agent(sweep_id, function=main, count=10)