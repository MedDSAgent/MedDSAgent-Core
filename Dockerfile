# ============================================================
# Stage 1 — TypeScript build
# ============================================================
FROM node:22-slim AS builder

# better-sqlite3 native addon requires Python + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        make \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
RUN npm run build

# ============================================================
# Stage 2 — Runtime image
# ============================================================
FROM node:22-slim AS runtime

# Python runtime for the data-science subprocess workers
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Compiled JS + prompts
COPY --from=builder /app/dist         ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Python worker package (spawned as subprocess by WorkerProcess)
COPY python_worker/ ./python_worker/

# Install Python worker dependencies (requirements.txt is the source of truth)
RUN pip3 install --no-cache-dir --break-system-packages \
        -r python_worker/requirements.txt

# Workspace volume — mount a host directory here at runtime
RUN mkdir -p /workspace
VOLUME /workspace

ENV WORK_DIR=/workspace
ENV HOST=0.0.0.0
ENV PORT=7842
# Use the system python3 for the Python worker subprocess
ENV MEDDS_PYTHON_BIN=python3

EXPOSE 7842

# Node 22 has built-in fetch; no extra tooling needed for healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -qO- http://localhost:7842/health || exit 1

CMD ["node", "dist/cli/index.js", "serve"]
