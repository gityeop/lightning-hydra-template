# src/data/sts_datamodule.py

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd


class STSDataset(Dataset):
    def __init__(self, sentence_pairs, scores, tokenizer, max_length):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        sentences = self.sentence_pairs[idx]
        score = self.scores[idx]

        # Tokenize the pair of sentences
        inputs = self.tokenizer(
            text=sentences[0],
            text_pair=sentences[1],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        return inputs, score


class STSDataModule(LightningDataModule):
    def __init__(
        self, plm_name, train_path, val_path, test_path, batch_size, max_length
    ):
        super().__init__()
        self.plm_name = plm_name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)

    def prepare_data(self):
        # Download or process data if necessary
        pass

    def setup(self, stage=None):
        # Load data
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)

            self.train_dataset = STSDataset(
                sentence_pairs=train_df[["sentence_1", "sentence_2"]].values.tolist(),
                scores=train_df["label"].values.tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

            self.val_dataset = STSDataset(
                sentence_pairs=val_df[["sentence_1", "sentence_2"]].values.tolist(),
                scores=val_df["label"].values.tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

        if stage == "test" or stage is None:
            test_df = pd.read_csv(self.test_path)

            self.test_dataset = STSDataset(
                sentence_pairs=test_df[["sentence_1", "sentence_2"]].values.tolist(),
                scores=test_df["label"].values.tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
