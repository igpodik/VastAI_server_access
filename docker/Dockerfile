FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# lightfm требует компиляции; conda-forge предоставляет готовый wheel
RUN conda install -y -c conda-forge lightfm=1.17 \
    && conda clean -afy

COPY docker/requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-train.txt \
    && rm /tmp/requirements-train.txt
