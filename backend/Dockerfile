FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/temp

# Set permissions
RUN chmod +x /app/start.sh 2>/dev/null || echo "start.sh not found, will create"

# Create start script
RUN echo '#!/bin/bash\n\
echo "🏥 Starting Medical Voice Assistant Backend..."\n\
echo "Environment: $ENVIRONMENT"\n\
echo "Port: $PORT"\n\
echo "Using Gemini: $USE_GEMINI"\n\
echo "Max Latency: $MAX_LATENCY_MS ms"\n\
echo "Primary TTS: $PRIMARY_TTS"\n\
echo ""\n\
python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --reload=${DEBUG:-false}\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["./start.sh"]
