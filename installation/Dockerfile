FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
COPY requirements_update.txt .
RUN  apt-get update 
RUN  apt-get install -y
RUN  apt install git -y
RUN  apt install -y python3.10
RUN  apt install -y python3-pip
RUN  pip3 install --no-cache-dir -r requirements_update.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libxrender-dev
