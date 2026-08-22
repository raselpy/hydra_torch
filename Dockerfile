FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Force stdout/stderr to flush immediately instead of block-buffering.
# Without this, logs (epoch progress, etc.) can appear to "hang" for long
# stretches when running under docker compose (non-TTY), then arrive all
# at once in a delayed burst — training is actually running fine the whole
# time, it's purely a buffering display issue.
ENV PYTHONUNBUFFERED=1

# git is required by DVC's scmrepo backend, even for local-only remotes
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install project deps first — layer-cached separately from later source
# changes, so editing main.py etc. doesn't force a full dependency reinstall
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir dvc

# Now copy the rest of the project
COPY main.py params.yaml dvc.yaml ./
COPY configs ./configs

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]