import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pytorch_lightning as pl
import en_core_web_sm
from lightning_get_loader import get_dataset, MyCollate
import torch.optim as optim
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

class pl_version(pl.LightningModule):

#======================================================
# MODEL(S)

    def __init__(self,
                embed_size,
                hidden_size, 
                vocab_size,
                num_layers,
                root_folder,
                annotation_file,
                dataset):

        super(pl_version, self).__init__()
        self.dataset = dataset

        # Encoder CNN
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear1 = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Decoder LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):

        # Encoder CNN
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear1(features))

        # Decoder LSTM
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear2(hiddens)

        return outputs

#     def caption_image(self, image, vocabulary, max_length=50):
#         result_caption = []

#         with torch.no_grad():
#             x = self.encoderCNN(image).unsqueeze(0)
#             states = None

#             for _ in range(max_length):
#                 hiddens, states = self.decoderRNN.lstm(x, states)
#                 output = self.decoderRNN.linear2(hiddens.squeeze(0))
#                 predicted = output.argmax(1)
#                 result_caption.append(predicted.item())
#                 x = self.decoderRNN.embed(predicted).unsqueeze(0)

#                 if vocabulary.itos[predicted.item()] == "<EOS>":
#                     break

#         return [vocabulary.itos[idx] for idx in result_caption]

#======================================================

#======================================================
# DATA
#    def train_dataloader(self):
#        return DataLoader(dataset = self.dataset,
#        batch_size = self.batch_size,
#        pin_memory = True,
#        num_workers = self.num_workers,
#        shuffle = self.shuffle,
#        collate_fn = MyCollate(pad_idx = self.pad_idx),
#        )


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
#======================================================
