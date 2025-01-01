import torch
import numpy as np
import wandb
import sys
import signal
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp
from scripts.train_modules.train_helpers import nsd


# DELETE
def log_metrics_to_wandb(job_type, metrics, wandb_logger):
    """
    Log metrics and visualizations to wandb.
    """
    print('++++++++++++++LOGGING METRICS TO W&B++++++++++++++')
    # Extract values from metrics
    water_accuracy = metrics["tps"] / (metrics["tps"] + metrics["fns"])
    precision = smp.metrics.precision(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()
    recall = smp.metrics.recall(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()
    iou = smp.metrics.iou_score(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()

    # water_accuracy = metrics["tps"] / (metrics["tps"] + metrics["fns"]) if metrics["tps"] + metrics["fns"] > 0 else 0.0
    # precision = metrics.get("precision", smp.metrics.precision(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean())
    # recall = metrics.get("recall", smp.metrics.recall(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean())
    # iou = metrics.get("iou", smp.metrics.iou_score(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean())


    # Log metrics
    wandb_logger.log_metrics({
        f"{job_type}flood_not_missed": water_accuracy,
        f"{job_type}iou": iou,
        f"{job_type}precision": precision,
        f"{job_type}recall": recall,
        f"{job_type}nsd_avg": metrics["nsd_avg"],
    })

    # Log images
    # for i, (logit, mask) in enumerate(zip(logits, masks)):
    #     pred_image = (logit[0] > 0.5).cpu().numpy()
    #     gt_image = mask[0].cpu().numpy()

    #     wandb_logger.experiment.log({
    #         f"sample_{i}_prediction": wandb.Image(pred_image, caption="Prediction"),
    #         f"sample_{i}_ground_truth": wandb.Image(gt_image, caption="Ground Truth")
    #     })

# FOR INFERENCE / COMPARISON FN
def calculate_metrics(logits, masks, metric_threshold):
    """
    Calculate TP, FP, FN, TN, and related metrics for a batch of predictions.
    """
    # Initialize accumulators
    tps, fps, fns, tns = [], [], [], []
    nsds = []

    for logit, mask in zip(logits, masks):
        # metric predictions
        tp, fp, fn, tn = smp.metrics.get_stats(
            logit, mask.long(), mode='binary', threshold=metric_threshold
        )
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)

        # Compute Normalized Spatial Difference (NSD)
        nsd_value = nsd(
            logit[0].cpu().numpy() > metric_threshold,
            mask[0].cpu().numpy().astype(bool),
        )
        nsds.append(nsd_value)

    # Aggregate metrics
    tps = torch.vstack(tps).sum()
    fps = torch.vstack(fps).sum()
    fns = torch.vstack(fns).sum()
    tns = torch.vstack(tns).sum()

    return {
        "tps": tps,
        "fps": fps,
        "fns": fns,
        "tns": tns,
        "nsd_avg": np.mean(nsds)
    }
        

def handle_interrupt(signal, frame):
    '''
    usage: signal.signal(signal.SIGINT, handle_interrupt)
    '''
    print("Interrupt received! Cleaning up...")
    # Add any necessary cleanup code here (e.g., saving model checkpoints)
    sys.exit(0)


def loss_chooser(loss_name, alpha=0.25, gamma=2.0, bce_weight=0.5):
    '''
    MUST ADDRESS:
    - CLASS IMBALANCE: Handled by Focal Loss and BCE-Dice combinations.
    - HIGH RECALL: Dice loss emphasizes overlap, helping improve recall.
    - BOUNDARY ACCURACY: Dice and Focal Loss focus on boundary regions.

    Parameters:
    - loss_name: Name of the desired loss function.
    - alpha: Weighting factor for the minority class (used in Focal Loss).
    - gamma: Modulating factor for hard examples (used in Focal Loss).
    - bce_
    '''

    if 'focal' in loss_name:
        print(f'---alpha: {alpha}, gamma: {gamma}---')  
    torch_bce = torch.nn.BCEWithLogitsLoss()
    smp_bce =  smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode='binary')
    focal = smp.losses.FocalLoss(mode='binary', alpha=alpha, gamma=gamma)
    # Adjust alpha if one class dominates or struggles.
    # Adjust gamma to fine-tune focus on hard examples

    def bce_dice(preds, targets):
        return bce_weight * smp_bce(preds, targets) + (1 - bce_weight) * dice(preds, targets)

    if loss_name == "torch_bce":
        return torch_bce        
    if loss_name == "smp_bce":
        return smp_bce
    if loss_name == "focal": # no weighting
        return focal
    if loss_name == "dice":
        return dice
    if loss_name == "bce_dice":
        return bce_dice

    elif loss_name == "tversky": # no weighting
        return smp.losses.TverskyLoss()
    elif loss_name == "jakard":
        return smp.losses.JaccardLoss() # penalize fp and fn. use with bce

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    

def wandb_initialization(job_type, repo_path, project, dataset_name, run_name, train_list, val_list, test_list, wandb_config):
    """
    Initialize W&B and return a WandbLogger for PyTorch Lightning.
    Handles dataset artifacts for 'train', 'test', and 'reproduce' jobs.
    """
    # Set default parameters
    mode = 'online'

    # Update parameters based on job type
    if job_type == "reproduce":
        artifact_dataset_name = f'unosat_emergencymapping-United Nations Satellite Centre/{project}/{dataset_name}/{dataset_name}'
    elif job_type == "debug":
        mode = 'disabled'


    # Initialize W&B run
    run = wandb.init(
        project=project,
        job_type=job_type,
        name=run_name,
        config=wandb_config,
        mode=mode,
        dir=repo_path / "4results",
    )

    

    if job_type != 'reproduce':
        # Create and log dataset artifact
        data_artifact = wandb.Artifact(
            dataset_name,
            type="dataset",
            description=f"{dataset_name} - 12 events",
            metadata={
                "train_list": str(train_list),
                "val_list": str(val_list),
                "test_list": str(test_list),
            },
        )
        # Add references
        train_list_path = str(train_list).replace("\\", "/")
        train_list_uri = f"file:////{train_list_path}"
        data_artifact.add_reference(train_list_uri, name="train_list")
        run.log_artifact(data_artifact, aliases=[ dataset_name])

    elif job_type == 'reproduce':
        # Retrieve dataset artifact
        data_artifact = run.use_artifact(artifact_dataset_name)
        metadata_data = data_artifact.metadata
        print(">>> Current Artifact Metadata:", metadata_data)

        # Update dataset paths
        train_list = Path(metadata_data.get('train_list', train_list))
        val_list = Path(metadata_data.get('val_list', val_list))
        test_list = Path(metadata_data.get('test_list', test_list))

        # Warn if any path is missing
        if not train_list.exists():
            print(f"Warning: train_list path {train_list} does not exist.")
        if not val_list.exists():
            print(f"Warning: val_list path {val_list} does not exist.")
        if not test_list.exists():
            print(f"Warning: test_list path {test_list} does not exist.")

    # Return WandbLogger for PyTorch Lightning
    return WandbLogger(experiment=run)

              
def job_type_selector(job_type):

    train, test,  debug = False, False, False

    if job_type == "train":
        train = True
    elif job_type == "test":
        test = True
    elif job_type == "debug":
        debug = True

    return train, test, debug

    

