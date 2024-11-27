import streamlit as st
import torch
from transformers import AutoTokenizer
import mlflow
import pandas as pd
import plotly.express as px
from pathlib import Path

from model import ModelTrainer, ModelConfig

class StreamlitApp:
    def __init__(self):
        self.config = ModelConfig()
        self.load_model()
        
    def load_model(self):
        model_path = Path("output/model")
        if not model_path.exists():
            st.error("Model not found! Please train the model first.")
            return
            
        self.trainer = ModelTrainer(self.config)
        self.trainer.model.load_state_dict(
            torch.load(model_path / "pytorch_model.bin")
        )
        self.trainer.model.eval()
        
    def predict(self, text: str) -> dict:
        inputs = self.trainer.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.trainer.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.trainer.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
        return {
            "positive": float(probs[0][1]),
            "negative": float(probs[0][0])
        }

def main():
    st.title("Sentiment Analysis Demo")
    
    app = StreamlitApp()
    
    # Text input
    text = st.text_area("Enter text to analyze:", "This movie was amazing!")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            results = app.predict(text)
            
            # Display results
            st.subheader("Results")
            fig = px.bar(
                x=["Positive", "Negative"],
                y=[results["positive"], results["negative"]],
                title="Sentiment Probabilities"
            )
            st.plotly_chart(fig)
            
            # Display confidence
            confidence = max(results.values())
            sentiment = "positive" if results["positive"] > results["negative"] else "negative"
            st.write(f"Prediction: {sentiment.upper()} with {confidence:.2%} confidence")
    
    # Display MLflow experiments
    st.subheader("Training History")
    try:
        mlflow_runs = mlflow.search_runs()
        if not mlflow_runs.empty:
            st.line_chart(
                mlflow_runs[["metrics.train_loss", "metrics.val_loss"]]
            )
    except Exception as e:
        st.warning("No training history available yet.")

if __name__ == "__main__":
    main()