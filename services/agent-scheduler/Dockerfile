FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools for sentence-transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        libffi-dev \
        libssl-dev \
        tini && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared packages
COPY packages/ packages/

# Copy service code
COPY services/agent-scheduler/ .

# Create non-root user
RUN useradd -m -u 1000 agentuser && \
    chown -R agentuser:agentuser /app

USER agentuser

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "main.py"]