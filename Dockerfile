FROM nvidia/cuda:10.1-runtime-ubuntu18.04
COPY installation/requirements.txt .
RUN  apt update 
RUN  apt-get install -y python3.6 
RUN  apt install -y python3-pip
RUN  pip3 install --no-cache-dir -r requirements.txt
