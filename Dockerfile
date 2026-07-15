# ============================================================
# Stage 1 — TypeScript build
# ============================================================
# Node 24: the DB layer uses the built-in node:sqlite, which is only
# available unflagged from Node 24 onward.
FROM node:24-slim AS builder

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
FROM node:24-slim AS runtime

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

# R worker source. NOTE: this image does not install R, so `language: "r"` will
# fail here with "Rscript not found". The source ships anyway (it is a few KB) so
# that an image derived from this one only needs to add R + the jsonlite package:
#   FROM <this image>
#   RUN apt-get update && apt-get install -y r-base \
#    && Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
COPY r_worker/ ./r_worker/

# Workspace volume — mount a host directory here at runtime
RUN mkdir -p /workspace
VOLUME /workspace

ENV WORK_DIR=/workspace
ENV HOST=0.0.0.0
ENV PORT=7842
# Use the system python3 for the Python worker subprocess
ENV MEDDS_PYTHON_BIN=python3

EXPOSE 7842

# wget is installed above for the healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -qO- http://localhost:7842/health || exit 1

CMD ["node", "dist/cli/index.js", "serve"]
