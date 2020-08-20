import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from lightning_get_loader import get_dataset
from lightning_model import pl_version
import pytorch_lightning as pl

import os
d = os.path.dirname(os.getcwd())
root_folder = d+"/project/data/tiny_coco/images"
annotation_file=d+"/project/data/tiny_coco/captions.txt"

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset, _ = get_dataset(root_folder, annotation_file, transform)

# Hyperparameters
embed_size = 5
hidden_size = 5
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 1

# initialize model, loss etc
model = pl_version(embed_size, hidden_size, vocab_size, num_layers, root_folder, annotation_file, dataset, learning_rate)

trainer = pl.Trainer(gpus = 10, num_nodes = 5)

trainer.fit(model)


