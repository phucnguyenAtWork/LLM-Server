# ── FINA AI Brain — Dockerfile ────────────────────────────────────────────────
# Base: PyTorch + CUDA 12.1 (matches the Windows dev environment)
# GPU access at runtime: docker run --gpus all ...

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (exclude model files — mounted as volume)
COPY api.py .
COPY train.py .
COPY chat.py .
COPY generate_hybrid.py* ./
COPY nlp/        ./nlp/
COPY categorizer/ ./categorizer/
COPY forecasting/ ./forecasting/
COPY rag/         ./rag/

# Create directories that will be populated via volumes
RUN mkdir -p models chroma_db financial_qwen_native_v1

# Expose the API port
EXPOSE 8105

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8105/ || exit 1

# Start the API
CMD ["python", "api.py"]
