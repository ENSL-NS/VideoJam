FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set up environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    python3-pip \
    lsof \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy opencv-python-headless aiohttp psutil

# Start Python script
WORKDIR /cfv

#COPY somthing .
#CMD ["python3", "python_script.py"]