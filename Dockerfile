FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        # imaging libs required by docling
        libgl1 \
        libglib2.0-0 \
        # R runtime required by rpy2
        r-base \
        r-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch BEFORE docling so the resolver does not pull
# in the large CUDA build (~2 GB) when docling lists torch as a dependency.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package with all runtime extras
COPY . .
RUN pip install --no-cache-dir ".[server,docling,r]"

# Workspace is mounted as a volume at runtime
RUN mkdir -p /workspace
VOLUME /workspace

ENV WORK_DIR=/workspace
ENV HOST=0.0.0.0
ENV PORT=7842

EXPOSE 7842

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7842/health')"

CMD ["medds-server"]
