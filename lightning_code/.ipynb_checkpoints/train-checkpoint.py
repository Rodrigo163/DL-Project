import pytorch_lightning as pl
from lightning_test import LightningMNISTClassifier

# train.py
def main():
    model = LightningMNISTClassifier()
    trainer = pl.Trainer(gpus = 2, num_nodes = 32,  auto_select_gpus = True, profiler = True, distributed_backend=’ddp’, early_stop_callback=True)
    trainer.fit(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # TRAIN
    main()