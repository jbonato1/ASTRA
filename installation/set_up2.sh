#!/bin/bash

ENV_NAME="AstroSSS"

echo $ENV_NAME
STDIR=$PWD
PARDIR="$(dirname "$PWD")"
cd $PARDIR

virtualenv .$ENV_NAME
ENV_PATH="."$ENV_NAME"/bin/activate"
#echo $ENV_PATH
source $ENV_PATH

