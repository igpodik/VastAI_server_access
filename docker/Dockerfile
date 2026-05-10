FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

COPY docker/requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-train.txt \
    && rm /tmp/requirements-train.txt
