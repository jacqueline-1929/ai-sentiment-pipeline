import pytest
import torch
from app.model import ModelTrainer, ModelConfig, TextDataset

@pytest.fixture
def model_config():
    return ModelConfig(
        model_name="prajjwal1/bert-tiny",  # Tiny model for testing
        num_labels=2,
        batch_size=2,
        max_length=32,
        num_epochs=1
    )

@pytest.fixture
def sample_data():
    texts = [
        "This movie was great!",
        "Terrible waste of time.",
        "Amazing performance!",
        "I hated everything about it."
    ]
    labels = [1, 0, 1, 0]
    return texts, labels

def test_model_initialization(model_config):
    trainer = ModelTrainer(model_config)
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert isinstance(trainer.device, torch.device)

def test_data_preparation(model_config, sample_data):
    trainer = ModelTrainer(model_config)
    texts, labels = sample_data
    train_loader, val_loader, test_loader = trainer.prepare_data(texts, labels)
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

def test_model_forward_pass(model_config, sample_data):
    trainer = ModelTrainer(model_config)
    texts, labels = sample_data
    
    # Create a small batch
    dataset = TextDataset(texts[:2], labels[:2], trainer.tokenizer, model_config.max_length)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    
    # Move batch to device
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = trainer.model(**batch)
    
    assert outputs.logits.shape == (2, 2)  # (batch_size, num_labels)

def test_model_training(model_config, sample_data):
    trainer = ModelTrainer(model_config)
    texts, labels = sample_data
    train_loader, val_loader, _ = trainer.prepare_data(texts, labels)
    
    # Train for one epoch
    trainer.train(train_loader, val_loader)
    
    # Test model saving
    trainer.save_model("test_output/model")
    assert (Path("test_output") / "model").exists()