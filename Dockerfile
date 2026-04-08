FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment source
COPY env.py .
COPY graders.py .
COPY inference.py .
COPY openenv.yaml .
COPY app.py .
COPY README.md .

# Expose Hugging Face Spaces port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command — launches the Gradio demo app
CMD ["python", "app.py"]
