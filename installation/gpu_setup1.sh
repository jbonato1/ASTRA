#!/bin/bash 

#if [ "$EUID" -ne 0 ]; then
#	echo "Please run as root (with sudo)"
#	exit
#fi

SETUP_DIR="$PWD"
echo SETUP_DIR
#mkdir -p $SETUP_DIR
#cd $SETUP_DIR

# install python libraries
#sudo apt-get -y install python-numpy python-dev python-wheel python-mock python-matplotlib python-pip

# install cuda drivers
if [ ! -f "cuda_10.1.105_418.39_linux.run" ]; then 
	wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run 
fi
#chmod +x cuda_10.1.105_418.39_linux.run
#sudo sh cuda_10.1.105_418.39_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.1
#
echo "Setup requires a reboot to continue."
echo "The VM will reboot now. Login after it restarts and continue installation from part2."

#sudo reboot
