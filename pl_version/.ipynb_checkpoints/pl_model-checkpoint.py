import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pytorch_lightning as pl
import en_core_web_sm
from pl_get_loader import get_dataset, MyCollate
import torch.optim as optim
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

#======================================================
#Data Module
class CocoDataModule(pl.LightningDataModule):

    def __init__(self, dataset,
                        batch_size,
                        transform,
                        num_workers,
                        ):

        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

        self.root_folder = os.path.dirname(os.getcwd())+"../data/images"
        self.annotation_file = os.path.dirname(os.getcwd())+ "../data/Captiones.tsv"


    def train_dataloader(self):
        return DataLoader(dataset = self.dataset,
        batch_size = self.batch_size,
        pin_memory = True,
        num_workers = self.num_workers,
        shuffle = True,
        collate_fn = MyCollate(pad_idx = 0),
        )

#======================================================

#======================================================
# MODEL MODULE

class pl_version(pl.LightningModule):
    
    #MODELS:
    
    def __init__(self,
                embed_size,
                hidden_size,
                vocab_size,
                num_layers,
                dataset):

        super(pl_version, self).__init__()
        self.dataset = dataset
        
        # Encoder CNN
        self.resnet = models.resnet18(pretrained=True)
        self.modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_seq = nn.Sequential(*self.modules)

        # Decoder LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def encode(self, images):
        features = self.resnet_seq(images)
        return features.permute(0, 2, 3, 1)
        
    def decode(self, features, captions):
        embeds = self.dropout(self.embed(captions))
        embeds = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeds)
        return self.linear(hiddens)
    
    def forward(self, images, captions):

        features = self.encode(images)
        outputs = self.decode(features, captions)
        
        return outputs
    
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            #encoding
            x = self.encode(image).unsqueeze(0)
            states = None
            
            #decoding
            for _ in range(max_length):
                hiddens, states = self.lstm(x, states)
                output = self.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

#======================================================

#======================================================
# OPTIMIZER
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
#======================================================

#======================================================
# LOSS
    def cross_entropy_loss(self):
        return nn.CrossEntropyLoss(ignore_index=self.dataset.vocab.stoi["<PAD>"])

#======================================================

#======================================================
# TRAINING
    def training_step(self, batch, batch_idx):
        #Unloading batch
        imgs, captions = batch

        #Forward pass
        outputs = self.forward(imgs, captions[:-1])
        loss = self.cross_entropy_loss()
        loss = loss(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        result = pl.TrainResult(loss)
        return result
