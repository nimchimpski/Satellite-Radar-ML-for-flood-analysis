# segmentation_trianing_loop.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import nn
from .z.boundaryloss import BoundaryLoss

class Segmentation_training_loop(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore = ['model'])   
        self.boundary_loss = BoundaryLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        # Container to store validation results
        self.validation_outputs = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        image = image.to('cuda')
        mask = mask.to('cuda')
        logits = self.forward(image)
        logits = logits.softmax(1)
        total_loss = 0
        class_weight = torch.ones_like(mask)
        class_weight[mask == 0] = 0.2

        # b_loss = self.boundary_loss(logits, mask.long().squeeze(1))
        b_loss = F.cross_entropy(logits, mask.squeeze(1).long(), weight=torch.Tensor([0.2, 1]).cuda())
        total_loss = b_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = self._get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers().param_groups]
        return torch.Tensor([lr]).cuda()
    
    # def _get_current_lr(self):
    #     # Access the optimizer through self.optimizers() in PyTorch Lightning
    #     optimizer = self.optimizers()
    #     lr = [param_group["lr"] for param_group in optimizer.param_groups]
    #     return lr

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        logits = self.forward(image)
        logits = logits.softmax(1)
        
        flood_loss = F.cross_entropy(logits, mask.squeeze(1).long(), weight=torch.Tensor([0.2, 1]).cuda())
        tp, fp, fn, tn = smp.metrics.get_stats(logits[:, 1:], mask.long(), mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn)
        self.log('val_loss', flood_loss, on_epoch=True, on_step= True, prog_bar=True, logger=True)
        self.log('iou', iou.mean(), prog_bar=True)
        return {"loss": flood_loss, "iou": iou.mean()}
        
        # OLD WORKED
        # self.log('val_loss', flood_loss, prog_bar=True)
        # self.log('iou', iou.mean(), prog_bar=True)
        # return {"loss": flood_loss, "iou": iou.mean()}

    def test_step(self, batch, batch_idx):
        image, mask, dist = batch
        logits = self.forward(image)
        pd = logits.cpu().numpy()
        gt = mask.cpu().numpy()
        # test logic here
        return {"results": None}

    
    def configure_optimizers(self):
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
    


class Single_channel_training_loop(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore = ['model'])   
        self.boundary_loss = BoundaryLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()        # Container to store validation results
        self.validation_outputs = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        image, mask = batch

        # Debugging: Check unique values in the mask
        print("Unique values in mask:", torch.unique(mask))

        image = image.to('cuda')
        mask = mask.to('cuda')
        logits = self.forward(image)
        logits = logits.squeeze(1)

        # BCE loss for single-channel output
        flood_loss = self.bce_loss(logits, mask.float())        
        total_loss = flood_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = self._get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers().param_groups]
        return torch.Tensor([lr]).cuda()

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        logits = self.forward(image)
        logits = logits.squeeze(1)
        
        flood_loss = self.bce_loss(logits, mask.float())

        # Apply sigmoid for metrics
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(predictions, mask.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn)

        self.log('val_loss', flood_loss, on_epoch=True, on_step= True, prog_bar=True, logger=True)
        self.log('iou', iou.mean(), prog_bar=True)

        # Debugging: Check unique values in the mask
        print("Unique values in validation mask:", torch.unique(mask))

        return {"loss": flood_loss, "iou": iou.mean()}
        
        # OLD WORKED
        # self.log('val_loss', flood_loss, prog_bar=True)
        # self.log('iou', iou.mean(), prog_bar=True)
        # return {"loss": flood_loss, "iou": iou.mean()}

    def test_step(self, batch, batch_idx):
        image, mask = batch
        logits = self.forward(image)
        print('---logits:', logits.shape)
        logits = logits.squeeze(1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        return {'predictions': predictions, 'ground_truth': mask}

    
    def configure_optimizers(self):
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
    
    