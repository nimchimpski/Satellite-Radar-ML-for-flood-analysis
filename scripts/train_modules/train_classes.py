import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as Func
from torchvision import transforms
from torch import Tensor, einsum
from pytorch_lightning import seed_everything
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc

import torch
import torch.nn as nn
import rasterio
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb
import io
from scripts.train_modules.train_helpers import is_sweep_run


class FloodDataset(Dataset):
    def __init__(self, tile_list, tile_root, stage='train', inputs=None):
        with open(tile_list, 'r') as _in:
            sample_list = _in.readlines()
        self.sample_list = [t[:-1] for t in sample_list]
        
        if stage == 'train':
            self.tile_root = Path(tile_root, 'train')
        elif stage == 'test':
            self.tile_root = Path(tile_root, 'test')
        elif stage == 'val':
            self.tile_root = Path(tile_root, 'val')
        self.inputs = inputs

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.sample_list)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        # print(f'+++++++++++++++++++ get item')

        filename = self.sample_list[idx]
        file_path = Path(self.tile_root, filename)
        # Use rasterio to inspect layer descriptions and dynamically select layers
        with rasterio.open(file_path) as src:
            layer_descriptions = src.descriptions  # Get layer descriptions
            # print(f"Layer Descriptions: {layer_descriptions}")  # Debugging

            # Ensure descriptions are present
            if not layer_descriptions:
                raise ValueError(f"No layer descriptions found in {file_path}")

            # Dynamically select layers based on their descriptions
            hh_index = layer_descriptions.index('hh')  # Index of 'hh' layer
            mask_index = layer_descriptions.index('mask')  # Index of 'mask' layer

            # Read only the required layers
            hh = src.read(hh_index + 1)  # Rasterio uses 1-based indexing
            mask = src.read(mask_index + 1)

        # tile = tiff.imread(Path(self.tile_root, filename))

        # print tile info and shape
        # print(f"---Tile shape b4 permute: {tile.shape}")

        # Transpose to (C, H, W)
        # tile = torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1)  # Shape: (2, 256, 256)

        # print(f"---Tile shape after permute: {tile.shape}")

        # Ensure the tile has 2 channels
        # assert tile.shape[0] == 2, f"Unexpected number of channels: {tile.shape[0]}"    
        # Select channels based on `inputs` list position
        # input_idx = list(range(len(self.inputs)))
        # model_input = tile[input_idx, :, : ]  # auto select the channels
        # model_input = tile[:1, :, : ].clone()  # auto select the channels
        model_input = torch.tensor(hh, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # print(f"---model_input shape: {model_input.shape}")  # Should print torch.Size([batch_size, 2, 256, 256])  

        model_input = model_input.cuda() #???

        # EXTRACT MASK TO BINARY
        # print('---mask index:', self.inputs.index('mask'))
        # mask = tile[ self.inputs.index('mask'),:,: ]
        # mask = tile[ 1,:,: ].clone()
        # mask = mask.unsqueeze(0)  # Add a channel dimension
        # CONVERT TO TENSOR
        mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)  # Add a channel dimension
        mask = (mask > 0.5).float()

        # print(f"---mask shape: {mask.shape}")  # Should print torch.Size([batch_size, 1, 256, 256])

        assert mask.shape == (1, 256, 256), f"Unexpected mask shape: {mask.shape}"

        # Debugging: Check unique values in the mask
        # print("---Unique values in mask:", torch.unique(mask))

        # Combine HH and MASK into a single input tensor
        # input_tensor = torch.stack([model_input, mask], dim=0)  # Shape: (2, 256, 256)
        return [model_input, mask]
        # return model_input.float(), val_mask.float()


class Segmentation_training_loop(pl.LightningModule):

    def __init__(self, model, loss_fn, save_path, user_loss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.save_hyperparameters(ignore = ['model', 'loss_fn', 'save_path'])   
        # Container to store validation results
        self.validation_outputs = []
        self.dynamic_weights = False
        self.user_loss = user_loss

        print(f'---loss_fn: {loss_fn}')

    def forward(self, x):
        # print(f"---Input device in forward: {x.device}")
        try:
            # for name, param in self.model.named_parameters():
            #     print(f"Parameter {name} is on device: {param.device}")  # Debug each parameter

            x = self.model(x)  # Pass through the model
            # print(f"---Output device in forward: {x.device}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise
        return x

    def training_step(self, batch, batch_idx):
        # print(f'+++++++++++++++++++   training step') 
        job_type = 'train'

        images, masks = batch
        images, masks = images.to(self.device), masks.to(self.device)
        logits = self(images)
        loss_per_pixel = self.loss_fn(logits, masks)  
        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        assert logits.device == masks.device

        lr = self._get_current_lr()

        _, _, _, _= self.metrics_maker(logits, masks, job_type, loss, self.user_loss, lr)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(f'+++++++++++++    validation step')
        job_type = 'val'

        images, masks = batch
        images, masks = images.to(self.device), masks.to(self.device)
        logits = self(images)
        # print(f"---Validation Step {batch_idx}: logits shape={logits.shape}, masks shape={masks.shape}")
        self.validation_outputs.append({'logits': logits, 'masks': masks})  # Store outputs
        loss_per_pixel = self.loss_fn(logits, masks)
        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        assert logits.device == masks.device
        # Check if this is the last batch and save visualization
        val_dataloader = self.trainer.val_dataloaders
        total_batches = len(val_dataloader) 
        # print(f"---Total batches: {total_batches}")
        # print info about images, preds, masks

        if not is_sweep_run():
            if self.current_epoch == self.trainer.max_epochs - 1 and batch_idx < 2:
                # This is the last epoch
                self.log_combined_visualization(images, logits, masks, self.user_loss)
                if batch_idx == 1:
                    print(f'---used dynamic weights = {dynamic_weights}')
            # Save images if this is the best-performing model
            if loss == self.trainer.checkpoint_callback.best_model_score and batch_idx < 2:
                self.log_combined_visualization(images, logits, masks, self.user_loss)

        ioumean, _, recall, _  = self.metrics_maker(logits, masks, job_type,  loss, self.user_loss )

        return {"loss": loss, "recall": recall, "iou": ioumean,  'logits': logits, 'labels': masks}   

    def test_step(self, batch, batch_idx):
        # print(f'+++++++++++++    test step')
        job_type = 'test'

        images, masks = batch
        images, masks = images.to('cuda'), masks.to('cuda')
        logits = self(images)
        self.test_outputs.append({'logits': logits, 'masks': masks})  # Store outputs

        loss_per_pixel = self.loss_fn(logits, masks)
        loss, dynamic_weights = self.dynamic_weight_chooser(masks, loss_per_pixel, self.user_loss)
        assert logits.device == masks.device

        # print(f"---weighted_loss device: {weighted_loss.device}")

        # preds = (torch.sigmoid(logits) > 0.5).int() # BCE, 

        # CALCULATE METRICS
        ioumean, precisionmean, recallmean, f1mean = self.metrics_maker(logits, masks, job_type, loss, self.user_loss)
        # Determine if this is the last batch
        test_dataloader = self.trainer.test_dataloaders # First DataLoader
        total_batches = len(test_dataloader)
        # print(f"---Total batches: {total_batches}")
        if batch_idx < 2:
            # print('---batch_idx:', batch_idx)
            # print(f"---Saving test outputs for batch {batch_idx}")
            self.log_combined_visualization(images, logits, masks, self.user_loss)  # Log the last batch visualization

            # JUST PRINT ON FIRST BATCH
            if batch_idx == 1:
                print(f'---used dynamic weights = {dynamic_weights}')



        return {"iou": ioumean, "precision": precisionmean, "recall": recallmean, "f1": f1mean, "loss": loss, 'logits': logits, 'labels': masks}

   
    
    def _get_current_lr(self):
        # print(f'+++++++++++++    get current lr')
        lr = [x["lr"] for x in self.optimizers().param_groups]
        return torch.Tensor([lr]).cuda()

    
    def compute_dynamic_weights(self, mask):
        
        # print(f'+++++++++++++    compute dynamic weights')
        assert torch.unique(mask).tolist() in [[0], [1], [0, 1]], f"Unexpected mask values: {torch.unique(mask)}"

        flood_pixels = (mask == 1).sum().float()
        non_flood_pixels = (mask == 0).sum().float()
        total_pixels = flood_pixels + non_flood_pixels

        if total_pixels > 0:
            flood_weight = non_flood_pixels / total_pixels
            non_flood_weight = flood_pixels / total_pixels
        else:
            flood_weight = 1.0
            non_flood_weight = 1.0

        weights = torch.ones_like(mask).float()
        weights[mask == 0] = non_flood_weight
        weights[mask == 1] = flood_weight

        weights = weights.to('cuda')

        return weights
    
    def dynamic_weight_chooser(self, masks, loss, user_loss):
        # print(f'+++++++++++++    dynamic weight chooser')
        # print(f'---loss_fn: {self.loss_fn}')
        if user_loss in ['smp_bce', 'bce_dice']:

            # print(f'*********computing dynamic weights')
            weights = self.compute_dynamic_weights(masks)
            weights = weights.to('cuda')
            assert masks.device == weights.device
            dynamic = True
            return  (loss * weights).mean(), dynamic
        else:
            # print(f'---no dynamic weights')
            dynamic = False
            return loss.mean(), dynamic 
        
    # from sklearn.metrics import f1_score

    # def find_best_threshold(y_true, y_probs):
    #     thresholds = [i / 100 for i in range(100)]
    #     f1_scores = [f1_score(y_true.flatten(), (y_probs.flatten() >= t)) for t in thresholds]
    #     best_threshold = thresholds[f1_scores.index(max(f1_scores))]
    #     return best_threshold

    def configure_optimizers(self):
        print(f'+++++++++++++    configure optimizers')
        params = [x for x in self.model.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)

        # Return as a list of dictionaries with `scheduler` and `interval` specified
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step" if you want to step every batch
                "frequency": 1
            }
        }
    
    
    def log_combined_visualization(self, images, logits, masks, loss_name):
        """
        Visualizes input images, predictions, and ground truth masks side by side for a batch.
        """
        # Handle thresholding based on the loss name
        if loss_name in ['smp_bce', 'bce_dice']:
            # Threshold predictions for visualization
            preds = (torch.sigmoid(logits) > 0.5).int()
            print(f'---THRESHOLDING FOR VISUALISATION')
        elif loss_name in ['dice', 'focal']:
            # Use continuous probabilities for visualization
            preds = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported loss name: {loss_name}. Choose from: 'smp_bce', 'dice', 'focal', 'bce_dice'.")


        print(f'+++++++++++++    log combined visualization')
        print(f'---images shape: {images.shape[0]}') 
        assert images.ndim == 4, f"Expected images with 4 dimensions (B, C, H, W), got {images.shape}"
        assert preds.ndim == 4, f"Expected preds with 4 dimensions (B, C, H, W), got {preds.shape}"
        assert masks.ndim == 4, f"Expected masks with 4 dimensions (B, C, H, W), got {masks.shape}"
        max_samples = 50  # Maximum number of samples to visualize
        for i in range(min(images.shape[0], max_samples)):  # Loop through each sample in the batch

            # print(f"---Sample {i}")
            # Convert tensors to numpy
            image = images[i].squeeze().cpu().numpy()
            pred = preds[i].squeeze().cpu().numpy()
            mask = masks[i].squeeze().cpu().numpy()

            # combined = np.concatenate([image, pred, mask], axis=1)
            # plt.imshow(np.concatenate([image, pred, mask], axis=1), cmap="gray")
            # plt.title(f"Sample {i} | Input | Prediction | Ground Truth")
            # plt.show()

            # print(f"Image min: {image.min()}, max: {image.max()}")
            # print(f"Pred min: {pred.min()}, max: {pred.max()}")
            # print(f"Mask min: {mask.min()}, max: {mask.max()}")

            # Normalize images and masks if needed (e.g., scale to [0, 255])
            epsilon = 1e-6
            image = (image - image.min()) / (image.max() - image.min() + epsilon) * 255  # Normalize input image to [0, 255]
            pred *= 255  # Threshold predictions and scale to [0, 255]
            mask *= 255  # Scale ground truth mask to [0, 255]

            # Stack the images horizontally (side by side)
            combined = np.concatenate([image, pred, mask], axis=1)
            plt.imshow(np.concatenate([image, pred, mask], axis=1), cmap="gray")
            plt.title(f"Sample {i} | Input | Prediction | Ground Truth")
            # plt.show()

            # Log to WandB
            self.logger.experiment.log({
                f"Sample {i} Visualization": wandb.Image(
                    combined.astype(np.uint8),  # Convert to uint8 for display
                    caption=f"Sample {i} | Input | Prediction | Ground Truth"
                )
            })


            
    
    

    def on_validation_epoch_start(self):
        self.validation_outputs = []  # Reset the list for the new epoch

    def on_validation_epoch_end(self):
        """
        Compute AUC-PR only during the final validation epoch.
        """
        # if is_sweep_run():
        #     print(f'---in sweep mode so skipping auc-pr calculation')
        #     return
        # Check if this is the final epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            print(f"---Calculating AUC-PR for the final validation epoch: {self.current_epoch}")

            # Ensure validation_outputs has been populated
            if not self.validation_outputs:
                raise ValueError("---Validation outputs are empty. Check your validation_step implementation.")

            # Aggregate outputs from all validation batches
            all_logits = torch.cat([output['logits'] for output in self.validation_outputs], dim=0)
            all_labels = torch.cat([output['masks'] for output in self.validation_outputs], dim=0)

            if all_logits.numel() == 0 or all_labels.numel() == 0:
                raise ValueError("Validation outputs are empty. Ensure validation_step is properly implemented.")


            # Flatten logits and labels
            logits_np = all_logits.detach().cpu().numpy().flatten()
            labels_np = all_labels.detach().cpu().numpy().flatten()

            # Check for NaN or Inf values in logits_np and labels_np
            if not np.isfinite(logits_np).all():
                raise ValueError("---logits_np contains NaN or Inf values.")

            if not np.isfinite(labels_np).all():
                raise ValueError("---labels_np contains NaN or Inf values.")
            
                        # Count unique classes
            unique_classes, class_counts = np.unique(labels_np, return_counts=True)
            print(f"Unique classes: {unique_classes}, Counts: {class_counts}")
            
            # Raise an error if there's only one class
            if len(unique_classes) < 2:
                raise ValueError("---Precision-Recall curve requires at least two classes in the ground truth.")


            try:
                precision, recall, _ = precision_recall_curve(labels_np, logits_np)
                auc_pr = auc(recall, precision)
            except ValueError as e:
                print(f"---AUC-PR calculation failed: {e}")
                auc_pr = 0.0  # Default value for invalid AUC-PR

            # Log the final AUC-PR
            self.log('val_auc_pr', auc_pr, prog_bar=True, logger=True)
        else:
        #     print(f"---Skipping AUC-PR calculation for epoch: {self.current_epoch}")
            self.validation_outputs = []

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        # Ensure validation_outputs has been populated
        if not self.test_outputs:
            raise ValueError("---test outputs are empty. Check your test_step implementation.")
        # Aggregate outputs
        all_logits = torch.cat([output['logits'] for output in self.test_outputs], dim=0)
        all_labels = torch.cat([output['masks'] for output in self.test_outputs], dim=0)

        # Convert to NumPy arrays
        logits_np = all_logits.cpu().numpy().flatten()
        labels_np = all_labels.cpu().numpy().flatten()

        # Compute Precision-Recall and AUC
        precision, recall, _ = precision_recall_curve(labels_np, logits_np)
        auc_pr = auc(recall, precision)

        # Log AUC-PR for the test set
        self.log('auc_pr_test', auc_pr, prog_bar=True, logger=True)

    def metrics_maker(self, logits, masks, job_type, loss, user_loss, lr=None):
        # print(f'+++++++++++++    metrics maker')
        # print(f"---Job type: {job_type}")
        # Thresholded predictions for metrics like IoU, precision, recall
        if user_loss in ['smp_bce', 'bce_dice', 'focal']:
            preds = (torch.sigmoid(logits) > 0.5).int()
        elif user_loss in ['dice']:
            preds = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported loss name: {user_loss}. Choose from:  'smp_bce', 'dice', 'focal', 'bce_dice'.")

        # Compute metrics
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn)
        precision = smp.metrics.precision(tp, fp, fn, tn)
        recall = smp.metrics.recall(tp, fp, fn, tn)
        f1 = smp.metrics.f1_score(tp, fp, fn, tn)

        # Averages
        ioumean = iou.mean()
        precisionmean = precision.mean()
        recallmean = recall.mean()
        f1mean = f1.mean()

        assert loss is not None, f"Loss is None for {job_type} job" 

        # Logging
        if job_type == 'train':
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train_lr', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif job_type == 'val':
            self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        if not job_type == 'train':
            self.log(f'iou_{job_type}', ioumean, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'precision_{job_type}', precisionmean, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'recall_{job_type}', recallmean, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'f1_{job_type}', f1mean, prog_bar=True, on_step=False, on_epoch=True)
    

        return ioumean, precisionmean, recallmean, f1mean
'''
def log_combined_visualization_plt(self, preds, mask):
        print(f'+++++++++++++    log combined visualization plt')
        print(f"Global Step: {self.global_step}")
        # Convert tensors to numpy
        preds = preds.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        # Show input image - stacked in reverse order!
        ax.imshow(mask, cmap="Blues", alpha=1)  # Overlay ground truth
        ax.imshow(preds, cmap="Reds", alpha=0.5)  # Overlay predictions
        ax.axis("off")

        legend_elements = [
            Line2D([0], [0], color="red", lw=4, label="Predictions"),
            Line2D([0], [0], color="blue", lw=4, label="Ground Truth")
        ]

        ax.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=True)

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
    
        # Convert buffer to PIL image for WandB
        pil_image = Image.open(buf)

        pil_image.save(self.save_path / f"debug_visualization_{self.global_step}.png")  # Save the image locally
        # print("Visualization saved locally as debug_visualization.png")

        # Log to WandB
        # Log to WandB with a unique key
        self.logger.experiment.log({
            f"Combined Visualization Step {self.global_step}": wandb.Image(
                pil_image,
                caption=f"Step {self.global_step} | Input with Prediction and Ground Truth Overlay"
            )
        })
    
        # Close buffer
        buf.close()
'''
# MODELS

class UnetModel(nn.Module):
    def __init__(self,encoder_name='resnet34', in_channels=1, classes=1, pretrained=True):
        super().__init__()
        self.model= smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels, 
            classes=classes,
            activation=None)
        
        # Fix the first convolutional layer
        self.model.encoder.conv1 = nn.Conv2d(
            in_channels=1,     # Match your input channel count
            out_channels=64,   # Keep the same number of filters
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.model.decoder.final_conv = nn.Conv2d(
        in_channels=1,  # Use the appropriate number of input channels from the decoder
        out_channels=1,  # Single output channel for binary segmentation
        kernel_size=1
)

        # if pretrained:
        #     checkpoint_dict= torch.load(f'/kaggle/working/Fold={index}_Model=mobilenet_v2.ckpt')['state_dict']
        #     self.model.load_state_dict(load_weights(checkpoint_dict))
    def forward(self,x):
        x= self.model(x)
        return x
 
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, classes=1):
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

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=1, pretrained=True):
        super().__init__()
        # Load pretrained ResNet
        self.backbone = models.__dict__[encoder_name](pretrained=pretrained)
        
        # Modify the first convolution layer to accept the HH SAR band (in_channels=1)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer to output a single logit for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # Single logit output

    def forward(self, x):
        return self.backbone(x)

# ------------------- LOSS FUNCTION -------------------

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # ****ADDED****
        # Check if the input tensor `one_hot_gt` is in the correct format
        # If it's not, convert it to float32
        if one_hot_gt.dtype != torch.float32:
            one_hot_gt = one_hot_gt.float()

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt


        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
        
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss
   
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# COMBOS
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)
        self.dice_loss = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * focal + (1 - self.alpha) * dice
    
class BoundaryDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        return self.alpha * dice + (1 - self.alpha) * boundary
    
class SurfaceDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.surface_loss = SurfaceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        surface = self.surface_loss(logits, targets)
        return self.alpha * dice + (1 - self.alpha) * surface
