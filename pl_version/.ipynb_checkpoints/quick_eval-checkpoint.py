import torch
import torchvision.transforms as transforms
from PIL import Image
from pl_get_loader import get_dataset, print_examples
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_model import pl_version
import os



transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset, _ = get_dataset("../data/images",
                    "../data/Captiones.tsv",
                    transform)

# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1



model = pl_version.load_from_checkpoint(checkpoint_path= 'lightning_logs/version_688899/checkpoints/epoch=96.ckpt',  hidden_size = hidden_size, vocab_size = vocab_size, dataset = dataset, num_layers = num_layers)# embed_size = embed_size)

model.eval()

print_examples(model, dataset)