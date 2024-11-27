import torch
import numpy as np
import wandb


import segmentation_models_pytorch as smp
from scripts.train_modules.train_helpers import nsd



def log_metrics_to_wandb(metrics, wandb_logger, logits, masks):
    """
    Log metrics and visualizations to wandb.
    """
    # Extract values from metrics
    water_accuracy = metrics["tps"] / (metrics["tps"] + metrics["fns"])
    precision = smp.metrics.precision(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()
    recall = smp.metrics.recall(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()
    iou = smp.metrics.iou_score(metrics["tps"], metrics["fps"], metrics["fns"], metrics["tns"]).mean()

    # Log metrics
    wandb_logger.log_metrics({
        "water_accuracy": water_accuracy,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "nsd_avg": metrics["nsd_avg"],
    })

    # Log images
    for i, (logit, mask) in enumerate(zip(logits, masks)):
        pred_image = (logit[0] > 0.5).cpu().numpy()
        gt_image = mask[0].cpu().numpy()

        wandb_logger.experiment.log({
            f"sample_{i}_prediction": wandb.Image(pred_image, caption="Prediction"),
            f"sample_{i}_ground_truth": wandb.Image(gt_image, caption="Ground Truth")
        })

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


