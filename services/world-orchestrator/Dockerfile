FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared packages
COPY packages/ packages/

# Copy service code
COPY services/world-orchestrator/ .

# Create non-root user
RUN useradd -m -u 1000 worlduser && \
    chown -R worlduser:worlduser /app

USER worlduser

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "main.py"]