FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    openai-whisper==20231117 \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    numpy==1.24.3 \
    python-multipart==0.0.6

# Copy service code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start the service
CMD ["python", "whisper_service.py"]
