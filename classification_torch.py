import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from torch import concat, optim
from torch.optim.lr_scheduler import ExponentialLR, SequentialLR, LambdaLR
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT
from PIL import Image
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4

# bands = [0, 1, 2]
tile_size = 224
# orig_nsize = 512
# crop_size = (tile_size, tile_size)
# img_suffix = ".jpg"
# seg_map_suffix = "_gtmask.png"
ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True

experiment = "training-output"
project_dir = "/scratch/sp6559/xie-training"
work_dir = os.path.join(project_dir, experiment)
data_root = os.path.join(project_dir, "dataset","classification_dataset","images")
pretrained_weights_path = os.path.join(project_dir,"prithvi","Prithvi_100M.pt")
pretrained_config = os.path.join(project_dir,"prithvi","Prithvi_100M_config.yaml")

torch.backends.cudnn.benchmark = True


LEARNING_RATE = 1.3e-05
EXPONENTIAL_LR_GAMMA = 0.1
WARMUP_ITERS= 1500
MAX_STEPS = 10000


img_norm_cfg = dict(
    means=[
        775.2290211032589,
        1080.992780391705,
        1228.5855250417867,
        2497.2022620507532,
        2204.2139147975554,
        1610.8324823273745
    ],
    stds=[
        1281.526139861424,
        1270.0297974547493,
        1399.4802505642526,
        1368.3446143747644,
        1291.6764008585435,
        1154.505683480695,
    ],
)  # change the mean and std of all the bands

class PILToNumpyReflct(object):
    def __init__(self,  means, stds, to_float32=False, nodata=None, nodata_replace=0.0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.conversion_means = means
        self.conversion_stds = stds

    def __call__(self, pil_img_label):
        img = np.asarray( pil_img_label[0], dtype="uint16" )
        img = img.transpose(2,0,1)[[2,1,0]] # Put channels first and convert to BGR from RGB

        img[0,:]  = img[0,:] * ((self.conversion_means[0] + self.conversion_stds[0] * 2) / 255)
        img[1,:]  = img[1,:] * ((self.conversion_means[1] + self.conversion_stds[1] * 2) / 255)
        img[2,:]  = img[2,:] * ((self.conversion_means[2] + self.conversion_stds[2] * 2) / 255)

        # to channels last format
        # img = np.transpose(img, (1, 2, 0))
        if self.to_float32:
            img = img.astype(np.float32)

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)

        return img

class TorchNormalizeAndDuplicate(object):
    """permutes, Normalizes the image. duplicates last 3 channels

    It normalises a multichannel image using torch

    Args:
        mean (sequence): Mean values .
        std (sequence): Std values of 3 channels.
    """

    def __init__(self, means, stds, duplicate = False):
        self.means = means
        self.stds = stds
        self.duplicate = duplicate

    def __call__(self, img):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.duplicate:
            img = F.normalize(img, self.means[0:3], self.stds[0:3], False)
            img = concat((img, img))
        else:
            img = F.normalize(img, self.means, self.stds, False)

        return img

class Reshape(object):
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return img.reshape(self.new_shape)


data_transforms = {
    'train': transforms.Compose([
        PILToNumpyReflct(**img_norm_cfg, to_float32=image_to_float32),
        transforms.ToTensor(),
        TorchNormalizeAndDuplicate(**img_norm_cfg, duplicate = True),
        Reshape((6, num_frames, tile_size, tile_size)),
    ]),
    'valid': transforms.Compose([
        PILToNumpyReflct(**img_norm_cfg, to_float32=image_to_float32),
        transforms.ToTensor(),
        TorchNormalizeAndDuplicate(**img_norm_cfg, duplicate = True),
        Reshape((6, num_frames, tile_size, tile_size)),
    ]),
}



#region Dataset Loading
image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                                    batch_size=samples_per_gpu,
                                                    shuffle=True,
                                                    num_workers=num_workers),
                'valid': torch.utils.data.DataLoader(image_datasets['valid'], 
                                                    batch_size=samples_per_gpu,
                                                    shuffle=False,
                                                    num_workers=num_workers)
            }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes
#endregion


#region read model config
weights_path = pretrained_weights_path
checkpoint = torch.load(weights_path) #, map_location=device)
model_cfg_path = pretrained_config
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)
model_args, train_args = model_config["model_args"], model_config["train_params"]
# let us use only 1 frame for now (the model was trained on 3 frames)
model_args["num_frames"] = 1
means_list = train_args["data_mean"]
std_list = train_args["data_std"]
embed_dim = model_args["embed_dim"]
#endregion

class ClassificationViT(nn.Module):
    def __init__(self,vit_args,checkpoint_loaded,embed_size):
        super(ClassificationViT, self).__init__()
        self.encoder_model = MaskedAutoencoderViT(**vit_args)
        del checkpoint_loaded['pos_embed']
        del checkpoint_loaded['decoder_pos_embed']
        _ = self.encoder_model.load_state_dict(checkpoint_loaded, strict=False)


        # self.pre_classifier = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(embed_size, 1)

    def forward(self, x):
        # https://discuss.pytorch.org/t/ensemble-of-five-transformers-for-text-classification/142719
        # Also look at B.1.1 Fine-tuning of ViT paper https://arxiv.org/pdf/2010.11929
        features, _, _ = self.encoder_model.forward_encoder(x, mask_ratio=0)
        reshaped_features = features[:, 0, :]
        # output = self.pre_classifier(reshaped_features)
        # output = torch.nn.Tanh(output)
        output = self.classifier(reshaped_features)
        return output

class ClassificationTrainingModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = self.model(x)
        loss = nnF.binary_cross_entropy_with_logits(x,y)
        self.train_accuracy(x, y)
        self.log('train/acc-step', self.train_accuracy, on_step=True, on_epoch=False)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = self.model(x)
        val_loss = nnF.binary_cross_entropy_with_logits(x,y) #combines sigmoid activation with ce loss
        self.valid_accuracy(x, y)
        self.log('valid/acc-step', self.valid_accuracy, on_step=True, on_epoch=False)
        self.log("val/loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        def warmup_routine(step):
            if step < WARMUP_ITERS:
                return  step / WARMUP_ITERS
            return WARMUP_ITERS ** 0.5 * step ** -0.5
        
        scheduler = (
            {
                "scheduler": LambdaLR(optimizer, warmup_routine),
                "interval": "step", #runs per batch rather than per epoch
                "frequency": 1,
                "name" : "learning_rate" # uncomment if using LearningRateMonitor
            }
        )
        # scheduler2 = ExponentialLR(optimizer, gamma=EXPONENTIAL_LR_GAMMA,min_lr=0.0)
        # scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        return [optimizer], [scheduler]

# if # has checkpoint file:     
#     crater_class_model = ClassificationTrainingModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
# else:
crater_class_model = ClassificationTrainingModule(ClassificationViT(model_args,checkpoint,embed_dim))

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="global_step",
    mode="max",
    dirpath=work_dir,
)
wandb_logger = WandbLogger(project="Martian Encoder")

# train model
trainer = L.Trainer(default_root_dir=work_dir, 
                    logger=wandb_logger,
                    max_steps=MAX_STEPS,
                    devices=num_workers,
                    accelerator="gpu",
                    callbacks=[checkpoint_callback, lr_monitor])
trainer.fit(model=crater_class_model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["valid"],
            ckpt_path="last",)

#region old train
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     # Create a temporary directory to save training checkpoints
#     with TemporaryDirectory() as tempdir:
#         best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

#         torch.save(model.state_dict(), best_model_params_path)
#         best_acc = 0.0

#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch}/{num_epochs - 1}')
#             print('-' * 10)

#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()  # Set model to training mode
#                 else:
#                     model.eval()   # Set model to evaluate mode

#                 running_loss = 0.0
#                 running_corrects = 0

#                 # Iterate over data.
#                 for inputs, labels in dataloaders[phase]:
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)

#                     # zero the parameter gradients
#                     optimizer.zero_grad()

#                     # forward
#                     # track history if only in train
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)
#                         _, preds = torch.max(outputs, 1)
#                         loss = criterion(outputs, labels)

#                         # backward + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # statistics
#                     running_loss += loss.item() * inputs.size(0)
#                     running_corrects += torch.sum(preds == labels.data)
#                 if phase == 'train':
#                     scheduler.step()

#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = running_corrects.double() / dataset_sizes[phase]

#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#                 # deep copy the model
#                 if phase == 'val' and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     torch.save(model.state_dict(), best_model_params_path)

#             print()

#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#         print(f'Best val Acc: {best_acc:4f}')

#         # load best model weights
#         model.load_state_dict(torch.load(best_model_params_path))
#     return model
#endregion