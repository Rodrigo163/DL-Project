import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from lightning_get_loader import get_dataset, get_loader
from lightning_model import pl_version
import pytorch_lightning as pl

import os
d = os.path.dirname(os.getcwd())
root_folder = d+"/project/data/images"
annotation_file=d+"/project/data/Captiones.tsv"

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

loader, dataset = get_loader(
                    root_folder,
                    annotation_file,
                    transform)

# Hyperparameters
embed_size = 5
hidden_size = 5
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 1

# initialize model, loss etc
model = pl_version(embed_size,
                    hidden_size,
                    vocab_size,
                    num_layers,
                    root_folder,
                    annotation_file,
                    dataset)

trainer = pl.Trainer(gpus = 2, num_nodes = 16,  auto_select_gpus = True, profiler = True, distributed_backend='ddp', early_stop_callback=True)

trainer.fit(model, loader)
