# Base image (small + stable)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install only REQUIRED system dependencies
# curl is needed because you are using HEALTHCHECK
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies (fast + no cache)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt


# Copy application code (respects .dockerignore)
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Railway exposes PORT dynamically, but EXPOSE is fine
EXPOSE 8000

# Start command
CMD ["python", "deploy.py"]