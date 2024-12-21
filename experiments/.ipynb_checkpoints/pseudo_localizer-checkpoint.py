import torch.nn as nn
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import gc, os
import yaml
import cv2
import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import monai
import albumentations as A
from sklearn.metrics import roc_auc_score
from cucim.core.operations.morphology import distance_transform_edt
from monai.losses import HausdorffDTLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,
    CenterSpatialCropd,  
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    RandShiftIntensityd,
    RandAffined,
    RandSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    ResizeWithPadOrCropd,    
    RandFlipd,
    RandZoomd,
    RandScaleIntensityd,
    SpatialPadd,
    ScaleIntensityRanged,
    MedianSmoothd,
    ToTensord,
    RandGridDistortiond,
    Rand3DElasticd,
    RandSimulateLowResolutiond,
    OneOf,
    RandomOrder
)
from torch.nn import BatchNorm3d
#from torchmetrics.classification import BinaryRecall
#from torchmetrics.classification import BinarySpecificity
# Stuff
import pickle
from sklearn.model_selection import train_test_split
import os, glob
import numpy as np
import wandb
import gc

# 'image_hydr', 'image_pneu'
transform_intensity_train = Compose(
  [
      EnsureChannelFirstD(keys=['image', 'mask'], channel_dim='no_channel'),
      #RandomOrder([
      RandScaleIntensityd(keys=['image'], factors=0.07, prob=0.09),
      RandShiftIntensityd(keys=['image'], offsets=14, prob=0.09),
      RandGaussianNoised(keys=['image'], mean=0, std=14, prob=0.09),
      #])
  ]
)
transform_intensity_test = Compose(
  [
        EnsureChannelFirstD(keys=['image', 'mask'], channel_dim='no_channel'),
  ]
)

transform_padd = Compose([
  DivisiblePadd(keys=['image', 'mask'], k=16, method='symmetric', mode='constant', value=0),
])

transform_train = Compose(
    [
        #RandZoomd(
        #    keys=['image', 'mask'], 
        #    mode=['bilinear', 'nearest'], 
        #    min_zoom=0.75, 
        #    max_zoom=1.25, 
        #    prob=0.5
        #),
        #
        RandAffined(
            keys=['image', 'mask'], 
            scale_range=0.2, 
            translate_range=25, 
            padding_mode='zeros', 
            mode=['bilinear', 'nearest'], 
            prob=0.5
        ),
        
        RandRotated(
            keys=['image', 'mask'], padding_mode='zeros',  mode=['bilinear', 'nearest'], 
            range_x=[-np.pi*15.0/180, np.pi*15.0/180], 
            range_y=[-np.pi*30.0/180, np.pi*30.0/180],
            range_z=[-np.pi*30.0/180, np.pi*30.0/180],
            prob=0.5
        ),
        #OneOf([
        #    RandGridDistortiond(
        #        keys=['image', 'mask'], 
        #        num_cells=4,
        #        prob=0.5,
        #        distort_limit=(-0.2, 0.2),
        #        mode=['bilinear', 'nearest']
        #    ),
        #    RandSimulateLowResolutiond(
        #        keys=['image'],
        #        prob=1.0, 
        #        downsample_mode='nearest',
        #        upsample_mode='bilinear',
        #        zoom_range=(0.5, 1.0)
        #    ),
        #    Rand3DElasticd(
        #        keys=['image', 'mask'],
        #        sigma_range=(5, 7),
        #        magnitude_range=(50, 150),
        #        prob=0.5,
        #        padding_mode='zeros',
        #        mode=['bilinear', 'nearest']
        #    ),
        #]),
        RandFlipd(keys=['image', 'mask'], prob=0.35, spatial_axis=0),
        RandFlipd(keys=['image', 'mask'], prob=0.35, spatial_axis=1),
        RandFlipd(keys=['image', 'mask'], prob=0.35, spatial_axis=2),
        
        #RandSpatialCropd(keys=['image', 'mask'], roi_size=[160, 224, 192], random_size=False),
        #SpatialPadd(keys=['image', 'mask'], spatial_size=[160, 224, 192], method='symmetric', mode='constant', value=0)  
        #RandSpatialCropd(keys=['image', 'mask'], roi_size=[128, 128, 128], random_size=False),
        #SpatialPadd(keys=['image', 'mask'], spatial_size=[128, 128, 128], method='symmetric', mode='constant', value=0)  
        #CenterSpatialCropd(keys=['image', 'seg'], roi_size=[128, 128, 128]),
    ]
)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

import torch

class DatasetAortaLocalizer(Dataset):
    def __init__(self, files=[], affine_transform=None, intensity_transform=None, padd_transform=None):
        self.affine_transform = affine_transform
        self.intensity_transform = intensity_transform
        self.padd_transform = padd_transform
        self.pairs = [x for x in files]
        self.clip = (-75, 280)
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        file = {
            'image': torch.from_numpy(np.load(self.pairs[idx][0])).float(),
            'mask': torch.from_numpy(np.load(self.pairs[idx][1]))
        }
        
        if self.intensity_transform:
            file = self.intensity_transform(file)
        
        file['image'] = torch.clip((file['image'] - self.clip[0])/(self.clip[1] - self.clip[0]), min=0, max=1)
        
        if self.affine_transform:
          file = self.affine_transform(file)
        
        if self.padd_transform:
          file = self.padd_transform(file)
        
        return file['image'], file['mask']

def dice_loss(net_output, target):
    return 1 - (2 * torch.sum(net_output * target)) / (torch.sum(net_output) + torch.sum(target) + 1e-6)

import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

def iou(net_output, target, class_label=1):
    net_out_ = (net_output == class_label)
    target_ = (target == class_label)
    return torch.sum(net_out_ * target_)/(torch.sum(torch.maximum(net_out_, target_)) + 1)

class Trainer:
    def __init__(self, config_name=''):
        # Get  files
        self.set_configs(config_name)
        self.set_wandb()
        self.set_net()
        self.set_opt_sched()
        self.set_loaders()
        # Initialize all essentials critetia and metrics
        self.SoftMax     = torch.nn.Softmax(dim=1)
        self.hausdorf    = HausdorffDTLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.topkloss    = TopKLoss()
        self.scaler      = GradScaler()
        self.best_acc    = 0.
        self.average_acc = 0.
        self.beta        = 0.95
    
    def criterion(self, outputs, masks, config):
        probability = self.SoftMax(outputs)
        
        loss_topk = self.topkloss(outputs, masks)
        config['loss_topk'].append(loss_topk.item())
        
        loss_dice = dice_loss(probability[:, 0], masks[:, 0]==0) + dice_loss(probability[:, 1], masks[:, 0]==1)
        loss_dice /= 2.0
        #loss_dice = dice_loss(probability[:, 1], masks[:, 0]==1)
        config['loss_dice'].append(loss_dice.item())
        
        loss_hausdorf = self.hausdorf(outputs, masks)
        config['loss_hausdorf'].append(loss_hausdorf.item())

        loss = 0.05*loss_hausdorf + (1-0.05)*loss_dice + loss_topk
        #loss = loss_dice
        config['loss'].append(loss.item())

        return loss
        
    def metrics(self, outputs, masks, config):
        probability = self.SoftMax(outputs)
        mask_pred = torch.argmax(probability, dim=1, keepdims=True)
        for i in range(masks.shape[0]):
            config['IoU'].append(iou(mask_pred[i], masks[i], class_label=1).item())

    def wandb_log_nums(self, config, name):
        wandb.log({
            f'Epoch'               : config['i_epoch'],
            f'{name} loss_topk'    : np.mean(config['loss_topk']),
            f'{name} loss'         : np.mean(config['loss']),
            f'{name} loss_dice'    : np.mean(config['loss_dice']),
            f'{name} loss_hausdorf': np.mean(config['loss_hausdorf']),
            f'{name} IoU'          : np.mean(config['IoU']),
            f'{name} Learning rate': get_lr(self.optimizer),
        })
        
    def model_selection(self, metrics):
        if np.mean(metrics['IoU']) > self.best_acc:
            self.best_acc = np.mean(metrics['IoU'])
            self.save_model(metrics, metrics['i_epoch'], self.model_name + f'_best')
            
    def test(self, i_epoch):
        # Test
        self.net.eval()
        config_log = self.get_log_config()
        # Inference
        with torch.no_grad():
            loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
            for batch_idx, (inputs, masks) in loop:
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                    
                #outputs = self.net(inputs)
                     
                outputs = sliding_window_inference(
                    inputs=inputs,         
                    roi_size=self.config['roi_size'],    
                    sw_batch_size=self.config['sw_batch_size'],
                    predictor=self.net,
                    overlap=self.config['overlap']
                )
                
                loss = self.criterion(outputs, masks, config_log)
                
                # Make backward step
                # Calculate and summary metrics
                self.metrics(outputs, masks, config_log)
                # LOOPA and PUPA
                loop.set_description(f"Epoch (Test)[{i_epoch}/{self.num_epochs}]")
                loop.set_postfix(IoU=np.mean(config_log['IoU']), loss=np.mean(config_log['loss']))

        config_log['i_epoch'] = i_epoch
        if self.wandb_use:
            self.wandb_log_nums(config_log, name='Test')
        
        # Save checkpoint.
        self.model_selection(config_log)
        

    def train(self, i_epoch):
        accum_iter = 4
        # Train
        config_log = self.get_log_config()
        self.net.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, masks) in loop:
            inputs, masks= inputs.to(self.device), masks.to(self.device)
            with torch.set_grad_enabled(True):
                #outputs = self.net(inputs)
                outputs = sliding_window_inference(
                    inputs=inputs,         
                    roi_size=self.config['roi_size'],    
                    sw_batch_size=self.config['sw_batch_size'],
                    predictor=self.net,
                    overlap=self.config['overlap']
                )
              
                loss = self.criterion(outputs, masks, config_log)/accum_iter
                loss.backward()
                
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                # Calculate and summary metrics
                self.metrics(outputs, masks, config_log)
                # LOOPA and PUPA
            
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(IoU=np.mean(config_log['IoU']), loss=np.mean(config_log['loss']))

        config_log['i_epoch'] = i_epoch
        if self.wandb_use:
            self.wandb_log_nums(config_log, name='Train')
        
    def fit(self):
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.test(i_epoch)
            #self.validate_product(i_epoch)
            self.scheduler.step()
        if self.wandb_use:
            self.run.finish()

    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.config['device'])

        self.net = monai.networks.nets.UNet(
            spatial_dims   = 3,
            in_channels    = self.config_net['in_channels'],
            out_channels   = self.config_net['out_channels'],
            channels       = self.config_net['channels'],
            strides        = self.config_net['strides'],
            kernel_size    = self.config_net['kernel_size'],
            up_kernel_size = self.config_net['up_kernel_size'],
            norm           = self.config_net['norm']
        )
        
        self.net = self.net.to(self.device)
        if self.wandb_use:
            wandb.watch(self.net, log_freq=100)

    def save_model(self, metrics, i_epoch, name):

        for key in metrics.keys():
            if isinstance(metrics[key], list):
                metrics[key] = np.mean(metrics[key])
        
        state = {
            'net'    : self.net.state_dict(), 
            'metrics': metrics,
            'config' : self.save_config,
            'epoch'  : i_epoch,
            'acc'    : self.best_acc
        }
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{name}.pth')
        print(f'Saved!:) Epoch[{i_epoch}], IoU = {np.mean(metrics["IoU"]):.04f}')
    
    def get_files(self):
        # Load split.
        files = torch.load(self.files_names[0])[0]
        return files['train'], files['test']
    
    def set_loaders(self):
        files_train, files_test = self.get_files()
        self.files_test = files_test
        # Create dataset and datalodaer for train
        train_ds = DatasetAortaLocalizer(
            files=files_train, 
            affine_transform=transform_train,
            intensity_transform=transform_intensity_train,
            padd_transform=transform_padd
        )
        self.train_loader = DataLoader(
            dataset=train_ds, 
            shuffle=True, 
            num_workers=self.config['num_workers_train'],
            batch_size=self.config['batch_size_train']
        )
        # Create dataset and dataloader for validation
        test_ds = DatasetAortaLocalizer(
            files=files_test, 
            affine_transform=None,
            intensity_transform=transform_intensity_test,
            padd_transform=transform_padd
        )
        self.test_loader = DataLoader(
            dataset=test_ds, 
            shuffle=False, 
            num_workers=self.config['num_workers_test'], 
            batch_size =self.config['batch_size_test']
        )
    
    def set_opt_sched(self):
        self.optimizer = torch.optim.AdamW(
            params       = self.net.parameters(),
            lr           = self.config_optimizer['learning_rate'],
            betas        = self.config_optimizer['betas'],
            eps          = self.config_optimizer['eps'],
            weight_decay = self.config_optimizer['weight_decay']
        )

        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer = self.optimizer, 
            step_size = self.config_scheduler['step_size'],
            gamma     = self.config_scheduler['gamma']
        )
        
        
    def load(self, filename, config_name=None):
        checkpoint = torch.load(f'./checkpoint/{filename}.pth', map_location='cpu')
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net         = self.net.to(self.device)
        self.best_acc    = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']        
        
        if not(config_name is None):
            if config_name == 'checkpoint':
                self.set_configs(config=checkpoint['config'])
            else:
                self.set_configs(config_name=config_name)
            
        if self.wandb_use:
            wandb.watch(self.net, log_freq=100)

    def set_configs(self, config_name='', config=None):
        if config is None:
            with open(f'./configs/{config_name}', 'r') as f:
                config = yaml.safe_load(f)
            
        self.config_net       = config['net']
        self.config_optimizer = config['optimizer']
        self.config_scheduler = config['scheduler']
        self.config           = config['hyperparams']
        self.files_names      = config['files']['names']
        self.save_config      = config
        
    def set_wandb(self):
        self.wandb_use = self.config['wandb_use']
        if self.wandb_use:
            wandb.login(key='')
            self.run = wandb.init(project=self.config['project'], config=self.save_config)   
            self.model_name = 'run_' + self.run.name + '_model'
        else:
            self.model_name = 'run_without_wandb_model'
            
        self.start_epoch = 0
        self.num_epochs  = self.config['epochs']
    
    def get_log_config(self):
        return {
            'i_epoch'       : 0,
            'loss'          : [],
            'loss_dice'     : [],
            'loss_topk'     : [],
            'loss_hausdorf' : [],
            'IoU'           : [],
            }