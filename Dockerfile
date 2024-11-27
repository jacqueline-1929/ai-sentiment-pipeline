# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports for services
EXPOSE 8501  # Streamlit
EXPOSE 8000  # FastAPI

# Create volume for model artifacts
VOLUME ["/app/models"]

# Command to run services
CMD ["sh", "-c", "streamlit run app/streamlit_app.py & uvicorn app.api:app --host 0.0.0.0 --port 8000"]