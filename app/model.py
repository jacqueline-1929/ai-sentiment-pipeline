import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 128
    random_state: int = 42
    max_samples: int = 10000  # Limit dataset size for faster training

def load_imdb_data(max_samples: int = 10000) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess the IMDB dataset from Hugging Face datasets.
    Returns preprocessed texts and labels.
    """
    logger.info("Loading IMDB dataset...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Convert to pandas for easier handling
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Subsample for faster training
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)
    
    # Clean text
    df['text'] = df['text'].apply(lambda x: x.strip())
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} samples")
    
    return df['text'].tolist(), df['label'].tolist()

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding=True, 
                                 max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)

class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels
        ).to(self.device)
        
        # MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("text_classification")

    def prepare_data(self, texts: List[str], labels: List[int]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=self.config.random_state
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=self.config.random_state
        )

        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        test_dataset = TextDataset(test_texts, test_labels, self.tokenizer, self.config.max_length)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                true_labels.extend(batch["labels"].cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        return total_loss / len(loader), accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.__dict__)
            
            for epoch in range(self.config.num_epochs):
                train_loss = self.train_epoch(train_loader, optimizer)
                val_loss, val_accuracy = self.evaluate(val_loader)
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )

    def save_model(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")
        
        # Save config
        with open(path / "config.yaml", "w") as f:
            yaml.dump(self.config.__dict__, f)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = ModelConfig(**cfg.model)
    trainer = ModelTrainer(config)
    
    # Load IMDB dataset
    texts, labels = load_imdb_data(max_samples=config.max_samples)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = trainer.prepare_data(texts, labels)
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    # Save the model
    trainer.save_model("output/model")
    
    # Final evaluation
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()