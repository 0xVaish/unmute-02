FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    transformers \
    fastapi \
    uvicorn \
    pydantic

# Install Dia TTS
RUN pip install --no-cache-dir git+https://github.com/nari-labs/dia.git

# Copy service files
COPY dia_service.py .

# Expose port
EXPOSE 8005

# Run the service
CMD ["python", "dia_service.py"]
