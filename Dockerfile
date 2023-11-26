FROM nvidia/cuda:12.3.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
	unzip \
        git \
        wget \
        ffmpeg \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install any python packages you need
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Set the working directory
WORKDIR /app
COPY . .

# Prepare the pretrained model
ENV PYTHONPATH "/app/MotionCorrection:${PYTHONPATH}"

RUN bash setup.sh

EXPOSE 8000

CMD ["python3", "main.py", "--host=0.0.0.0", "--port=8000"]
