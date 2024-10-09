# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04


LABEL authors="humanoid robot"


WORKDIR /app


COPY . /app


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/


RUN pip3 install --no-cache-dir --retries 5 -r /app/requirements.txt


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
