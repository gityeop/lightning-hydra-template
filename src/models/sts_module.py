# src/models/sts_module.py

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoModel
from torchmetrics.functional import pearson_corrcoef


class STSModel(LightningModule):
    def __init__(self, plm_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.plm = AutoModel.from_pretrained(plm_name)

        # Regression head
        self.regressor = nn.Linear(self.plm.config.hidden_size, 1)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        score = self.regressor(pooled_output)
        return score.squeeze(-1)  # (batch_size)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        scores = self(**inputs)
        loss = self.loss_fn(scores, labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        scores = self(**inputs)
        loss = self.loss_fn(scores, labels.float())
        pearson_corr = pearson_corrcoef(scores, labels.float())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pearson", pearson_corr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        scores = self(**inputs)
        loss = self.loss_fn(scores, labels.float())
        pearson_corr = pearson_corrcoef(scores, labels.float())
        self.log("test_loss", loss)
        self.log("test_pearson", pearson_corr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
