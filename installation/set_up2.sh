#!/bin/bash

ENV_NAME="envTest"

echo $ENV_NAME
STDIR=$PWD
PARDIR="$(dirname "$PWD")"
cd $PARDIR

virtualenv --python=python3.6 .$ENV_NAME
ENV_PATH="."$ENV_NAME"/bin/activate"
##echo $ENV_PATH
echo 'export PATH=/usr/local/cuda-10.1/bin:$PATH:' >> $ENV_PATH
echo 'export CUDADIR=/usr/local/cuda-10.1' >> $ENV_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH' >> $ENV_PATH
source $ENV_PATH
#
##check cuda installation comment if it is not necessary
#cp -a /usr/local/cuda-10.1/samples samples-10.1
#cd samples-10.1
#make -j 4
#cd ..
########
## installation of python packages
#
pip install -r requirements.txt
########
echo Installation complete
echo type source $ENV_PATH to activate the environment
echo type deactivate to exit from the environment
