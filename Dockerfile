# Built automatically by RunPod when you "Deploy from a GitHub repository".
# Base image already has torch 2.8.0 + CUDA 12.8, which matches fish-speech's pin.
FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev libsox-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# fish-speech (S2 Pro inference code)
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app \
    && pip install --no-cache-dir -e /app

# worker extras (also re-pins protobuf, which fish-speech's deps downgrade)
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# model lives on the network volume, mounted by RunPod at /runpod-volume
ENV CKPT=/runpod-volume/models/s2-pro-bn-bd
ENV COMPILE=1
ENV PORT=8000
ENV PORT_HEALTH=8000
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

COPY app/handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
