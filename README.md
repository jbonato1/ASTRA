# astrocyte_segmentation

This code has been developed with Python 3.6.7, Pytorch 1.1 with CUDA 10.1. All the packages are listed in the requirements.txt

## Installer
Installer can be used to install all the fundamental packages used in AstroSS if python 3.6.7 is already installed. If you need to
install python run:

sudo apt update
sudo apt-get install python3.6
sudo apt install python3-pip

With file gpu_setup1.sh in folder installation, cuda packages will be downloaded and installed
With file set_up2 all the necessary python packages will be installed and a virtual environment will be set-up


## Notebook

In this folder there are:

- Set_parameters: in this notebook it is described how to set all the hyperparameters of AstroSS using the training set 
- Training_Pipeline_PP: in this notebook it is described how to perform the preprocessing of training pipeline in AstroSS
- Training_Pipeline_Training_DNN: in this notebook it is described how to train the DNN
- Inference_Pipeline: In this notebook it is described how to perform AstroSS inference on the inference dataset
- CC_Pipeline


## AstroSS

### modules

- The modules used in Training and Inference pipelines are embedded in the libraries (.py files) 
- In model folder there is the DNN deceloped and used for benchmarking

### pipelines
In this folder there are pipelines used to perform experiments described in the paper 

## Zip_mask

- GT:In this folder there are the consunsus manual annotations of each dataset
- User_1: in this folder there are the Annotator-1 manual annotations of each dataset
- User_2: in this folder there are the Annotator-2 manual annotations of each dataset
- User_3:in this folder there are the Annotator-3 manual annotations of each dataset

## set 
In this folder there is the script to download each dataset and organize datasets in different folders

## weights
In this folder There is a scripts that can be used to download DNN weights used for benchmarking AstroSS