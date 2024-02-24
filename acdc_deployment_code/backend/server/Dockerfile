FROM nvidia/cuda:11.0.3-devel-ubuntu20.04 AS cuda-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_NAME="auto_heart"
ENV USER=root
ENV HOME=/root

USER root

ENV PYTHONPATH=/home/${USERNAME}/$APP_NAME:/home/${USERNAME}

RUN apt-get -y update
RUN apt-get  install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	build-essential \
        curl \
        gcc \
        g++ \
        python3.9-dev \
        python3-pip \
        python3.9 \
        libopenmpi-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0

## for fixing cuda key error
RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/cuda.list
RUN curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libgdcm-tools

RUN apt-get install -y unzip zip
RUN apt-get install -y python3-gdcm

RUN apt-get install -y poppler-utils
RUN apt-get install -y dcmtk
RUN apt-get install -y xvfb

RUN python3.9 -m pip install pip
RUN python3.9 -m pip install --upgrade pip

COPY  ./torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl ./torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl
RUN python3.9 -m pip install torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl


RUN python3.9 -m pip install vtk

RUN python3.9 -m pip install setuptools==45.2.0
RUN python3.9 -m pip install cython

RUN python3.9 -m pip install numpy matplotlib
RUN python3.9 -m pip install opencv-python==4.6.0.66

RUN python3.9 -m pip install pylibjpeg pylibjpeg-libjpeg
RUN python3.9 -m pip install pydicom pynetdicom SimpleITK==2.2.1 pandas scikit-image scikit-learn medpy nibabel


RUN python3.9 -m pip install pyyaml

RUN python3.9 -m pip install pymongo dnspython


##### flask and web
RUN python3.9 -m pip install flask==2.1.2
RUN python3.9 -m pip install flask_session
RUN python3.9 -m pip install flask_cors
RUN python3.9 -m pip install werkzeug
RUN python3.9 -m pip install uwsgi

# redis and queue
RUN python3.9 -m pip install redis
RUN python3.9 -m pip install albumentations==0.5.1
RUN python3.9 -m pip install torchvision





COPY ./src ./src
WORKDIR /src/

RUN ln -s /usr/bin/python3.9 /usr/bin/python
