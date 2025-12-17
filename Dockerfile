###################
# BUILD FOR PRODUCTION
###################
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serve_dropout_model.py .

# Copy the trained model artifacts
COPY artifacts/ ./artifacts/

# Expose the FastAPI port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "serve_dropout_model:app", "--host", "0.0.0.0", "--port", "8001"]
