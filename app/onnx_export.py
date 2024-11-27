import torch
from pathlib import Path
from model import ModelTrainer, ModelConfig

def export_to_onnx(trainer: ModelTrainer, output_path: Path) -> None:
    """Export the PyTorch model to ONNX format"""
    
    # Create dummy input
    dummy_input = {
        "input_ids": torch.ones(1, trainer.config.max_length, dtype=torch.long),
        "attention_mask": torch.ones(1, trainer.config.max_length, dtype=torch.long),
        "token_type_ids": torch.zeros(1, trainer.config.max_length, dtype=torch.long)
    }
    
    # Move to CPU for export
    trainer.model.cpu()
    dummy_input = {k: v.cpu() for k, v in dummy_input.items()}
    
    # Export the model
    torch.onnx.export(
        trainer.model,
        (dummy_input,),
        output_path / "model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=12
    )
    
    # Export tokenizer config
    trainer.tokenizer.save_pretrained(output_path / "tokenizer")
    
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    config = ModelConfig()
    trainer = ModelTrainer(config)
    
    # Load trained model
    model_path = Path("output/model")
    trainer.model.load_state_dict(
        torch.load(model_path / "pytorch_model.bin")
    )
    
    # Export to ONNX
    export_path = Path("output/onnx")
    export_path.mkdir(parents=True, exist_ok=True)
    export_to_onnx(trainer, export_path)