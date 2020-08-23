import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from pl_get_loader import get_dataset, get_loader
from pl_model import CaptionGenerator, CocoDataModule
import pytorch_lightning as pl
import os


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset, _ = get_dataset(
                        "../data/images",
                        "../data/Captiones.tsv",
                        transform)

train = pd.read_csv('../data/training_captions.tsv', sep = '\t')

# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
#num_epochs = 1

#Training parameters
num_nodes = 10
gpus = 2 #2 GPUs/node

#for loader
batch_size = 32
num_workers = int(num_nodes*2)


dm = CocoDataModule(train, dataset, batch_size, transform, num_workers)

# initialize model
model = CaptionGenerator(embed_size,
                    hidden_size,
                    vocab_size,
                    num_layers,
                    dataset)

trainer = pl.Trainer(gpus = gpus, num_nodes = num_nodes, max_epochs = 100, min_epochs = 50,  auto_select_gpus = True, profiler = True, distributed_backend='ddp', early_stop_callback=False)

trainer.fit(model, dm)
