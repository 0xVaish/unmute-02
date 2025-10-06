FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    transformers==4.36.2 \
    scipy==1.11.4 \
    numpy==1.24.3 \
    soundfile==0.12.1 \
    python-multipart==0.0.6 \
    git+https://github.com/suno-ai/bark.git

# Copy service code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Start the service
CMD ["python", "bark_service.py"]
