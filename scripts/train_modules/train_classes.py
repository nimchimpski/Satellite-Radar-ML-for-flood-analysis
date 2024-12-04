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
import torch
import torch.nn as nn
from pathlib import Path
import tifffile as tiff

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
        tile = tiff.imread(Path(self.tile_root, filename))

        # print tile info and shape
        # print(f"---Tile shape b4 permute: {tile.shape}")

        # Transpose to (C, H, W)
        tile = torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1)  # Shape: (2, 256, 256)

        # print(f"---Tile shape after permute: {tile.shape}")

        # Ensure the tile has 2 channels
        assert tile.shape[0] == 2, f"Unexpected number of channels: {tile.shape[0]}"    
        # Select channels based on `inputs` list position
        # input_idx = list(range(len(self.inputs)))
        # model_input = tile[input_idx, :, : ]  # auto select the channels
        model_input = tile[:1, :, : ].clone()  # auto select the channels
        model_input = torch.tensor(model_input, dtype=torch.float32)  

        # print(f"---model_input shape: {model_input.shape}")  # Should print torch.Size([batch_size, 2, 256, 256])  

        model_input = model_input.cuda() #???

        # EXTRACT MASK TO BINARY
        # print('---mask index:', self.inputs.index('mask'))
        # mask = tile[ self.inputs.index('mask'),:,: ]
        mask = tile[ 1,:,: ].clone()
        mask = mask.unsqueeze(0)  # Add a channel dimension
        # CONVERT TO TENSOR
        mask = torch.tensor(mask,dtype=torch.float32)
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

    def __init__(self, model):
        super().__init__()
        self.model = model

        self.save_hyperparameters(ignore = ['model'])   
        # self.boundary_loss = BoundaryLoss()
        # self.cross_entropy = nn.CrossEntropyLoss()


        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_fn = FocalLoss(alpha=0.25, gamma=2)
        # Container to store validation results
        self.validation_outputs = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # print(f'+++++++++++++++++++   training step') 
        image, mask = batch
        image, mask = image.to(self.device), mask.to(self.device)
        # print(f"---///Model device: {next(self.model.parameters()).device}")
        # print(f"---///Image device: {image.device}, Mask device: {mask.device}")
        # print(f"---Model running on device: {self.device}")
        # print(f"---^^^^^^^^Model device: {next(self.model.parameters()).device}")


        logits = self(image)
        total_loss = 0

        # Calculate weights dynamically based on the mask
        # Assuming 1 for flood class and 0.2 for non-flood class
        weights = torch.ones_like(mask).to(self.device)
        weights[mask == 0] = 0.2  # Less weight for non-flood
        weights[mask == 1] = 1.0  # Full weight for flood class

        # Compute Weighted BCE loss
        loss_per_pixel = self.loss_fn(logits, mask)  # Compute BCE per pixel
        weighted_loss = (loss_per_pixel * weights).mean()  # Apply weights and reduce

        self.log('train_loss', weighted_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        lr = self._get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return weighted_loss

        # total_loss = self.loss_fn(logits, mask)
        # self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # lr = self._get_current_lr()
        # self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # return total_loss

    def _get_current_lr(self):
        # print(f'+++++++++++++    get current lr')
        lr = [x["lr"] for x in self.optimizers().param_groups]
        return torch.Tensor([lr]).cuda()
    
    # def _get_current_lr(self):
    #     # Access the optimizer through self.optimizers() in PyTorch Lightning
    #     optimizer = self.optimizers()
    #     lr = [param_group["lr"] for param_group in optimizer.param_groups]
    #     return lr

    def validation_step(self, batch, batch_idx):
        # print(f'+++++++++++++    validation step')
        image, mask = batch
        image, mask = image.to(self.device), mask.to(self.device)
        logits = self.forward(image)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        logits = self.forward(image)
        logits = logits.softmax(1)

        # print(f"---Preds dtype: {preds.dtype}, unique values: {torch.unique(preds)}")
        # print(f"---Mask dtype: {mask.dtype}, unique values: {torch.unique(mask)}")

            # Compute Weighted BCE Loss
        weights = torch.ones_like(mask).to(self.device)
        weights[mask == 0] = 0.2
        weights[mask == 1] = 1.0

        loss_per_pixel = self.loss_fn(logits, mask)
        weighted_loss = (loss_per_pixel * weights).mean()  # Apply weights and reduce to scalar

        tp, fp, fn, tn = smp.metrics.get_stats(preds, mask.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn)

        self.log('val_loss', weighted_loss, on_epoch=True, on_step= True, prog_bar=True, logger=True)
        self.log('iou', iou.mean(), prog_bar=True)
        return {"loss": weighted_loss, "iou": iou.mean()}
        

    def test_step(self, batch, batch_idx):
        image, mask, dist = batch
        logits = self.forward(image)
        pd = logits.cpu().numpy()
        gt = mask.cpu().numpy()
        # test logic here
        return {"results": None}

    
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

# LOSS FUNCTION

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

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.alpha * bce + (1 - self.alpha) * dice