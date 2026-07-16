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

# No compiler is installed here on purpose: every Python and R dependency below
# comes from a prebuilt binary. Both installs are pinned to binary-only, so if a
# dependency ever stops shipping a wheel/binary the build fails loudly rather
# than silently reintroducing a ~330MB C/C++ toolchain into the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        wget \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# R for the R worker. Debian bookworm ships R 4.2.2, so use CRAN's own Debian
# repo for the current release. The key is fetched over HTTPS from the keyserver
# rather than port 11371, which is commonly firewalled.
#
# r-recommended supplies the packages bundled with a normal R install (survival,
# Matrix, lattice, …), which r-base-core omits. It matters beyond convenience:
# Posit Package Manager ships no binary for `survival`, so installing it via
# install.packages() below would fall back to a source build and fail for want of
# a compiler. Taking it from CRAN's apt repo gets a binary built for this exact R.
RUN wget -qO- "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7" \
      | gpg --dearmor -o /usr/share/keyrings/cran.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/debian bookworm-cran46/" \
      > /etc/apt/sources.list.d/cran.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        r-base-core \
        r-recommended \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Compiled JS + prompts
COPY --from=builder /app/dist         ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Python worker package (spawned as subprocess by WorkerProcess)
COPY python_worker/ ./python_worker/

# Install Python worker dependencies (requirements.txt is the source of truth).
# --only-binary=:all: — see the toolchain note above.
RUN pip3 install --no-cache-dir --break-system-packages --only-binary=:all: \
        -r python_worker/requirements.txt

# R worker package (spawned as subprocess by WorkerProcess)
COPY r_worker/ ./r_worker/

# Install R worker dependencies from Posit Package Manager, which serves prebuilt
# binaries for Debian bookworm. The HTTPUserAgent is what makes P3M return a
# binary instead of source — without it every package would compile, which would
# fail here since there is no compiler. type="binary" is not used because P3M
# delivers binaries through the source URL path on Linux.
RUN Rscript -e 'options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])), repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/bookworm/latest")); \
      pkgs <- c("jsonlite", "ggplot2", "dplyr", "tidyr", "data.table", "survival"); \
      install.packages(pkgs); \
      missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]; \
      if (length(missing)) { stop("R packages failed to install: ", paste(missing, collapse = ", ")) }'

# Workspace volume — mount a host directory here at runtime
RUN mkdir -p /workspace
VOLUME /workspace

ENV WORK_DIR=/workspace
ENV HOST=0.0.0.0
ENV PORT=7842
# Use the system python3 / Rscript for the worker subprocesses
ENV MEDDS_PYTHON_BIN=python3
ENV MEDDS_RSCRIPT_BIN=Rscript

EXPOSE 7842

# wget is installed above for the healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -qO- http://localhost:7842/health || exit 1

CMD ["node", "dist/cli/index.js", "serve"]
