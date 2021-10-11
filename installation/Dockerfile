FROM nvidia/cuda:10.1-devel-ubuntu18.04
COPY requirements.txt .
RUN  apt-get update 
RUN  apt-get install -y
RUN  apt install git -y
RUN  apt install -y python3.6 
RUN  apt install -y python3-pip
RUN  pip3 install --no-cache-dir -r requirements.txt

RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
