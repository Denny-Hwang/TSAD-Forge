# TSAD-Forge — PyTorch 2.6 + CUDA 12.4 (8GB VRAM 기준 기본 config)
# CPU 전용 사용 시: docker build --build-arg BASE=python:3.11-slim .
ARG BASE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

WORKDIR /workspace/TSAD-Forge

COPY pyproject.toml README.md LICENSE ./
COPY tsad_forge/ ./tsad_forge/

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["tsad-forge", "--help"]
