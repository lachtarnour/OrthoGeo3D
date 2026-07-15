ARG PYTORCH_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/workspace/data \
    OUTPUT_DIR=/workspace/outputs

WORKDIR /workspace/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && grep -vE '^(torch|torchvision|torchaudio)([=<>~! ].*)?$' requirements.txt > /tmp/requirements-runtime.txt \
    && python -m pip install -r /tmp/requirements-runtime.txt

COPY . .

RUN mkdir -p /workspace/data /workspace/outputs \
    && python -c "import torch, yaml, fpsample; print('OrthoTwin3D image build check OK'); print('torch', torch.__version__)"

CMD ["python", "-c", "import torch; print('OrthoTwin3D image ready'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"]
