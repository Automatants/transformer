import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim
import torch.nn.functional as F
import wandb
from model import GPT2
from dataset import TextDataset
from dataclasses import dataclass
import time
from torch.utils.data import DataLoader

@dataclass
class TrainingConfig:
    project: str = "gpt2"
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    tokens_per_batch: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    small_batch_size: int = 128
    batch_size: int = 1024

config = TrainingConfig()

train_dataset = TextDataset(
    tokens_per_batch=config.tokens_per_batch,
    batch_size=config.batch_size,
    split="train"
)

test_dataset = TextDataset(
    tokens_per_batch=config.tokens_per_batch,
    batch_size=config.batch_size,
    split="test"
)

val_dataset = TextDataset(
    tokens_per_batch=config.tokens_per_batch,
    batch_size=config.batch_size,
    split="validation"
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class GPT2Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT2()
        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss)
        end_time = time.time()
        self.log("time_per_step", end_time - start_time)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=3e-4)

model = GPT2Model()
wandb_logger = WandbLogger(project="gpt2")

trainer = pl.Trainer(
    max_epochs=10,
    gradient_clip_val=1.0,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
        )
    ]
)

trainer.fit(model, train_loader, val_loader)