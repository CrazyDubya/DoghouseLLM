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
COPY services/api-gateway/ .

# Create non-root user
RUN useradd -m -u 1000 gatewayuser && \
    chown -R gatewayuser:gatewayuser /app

USER gatewayuser

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "main.py"]