#!/bin/bash

# Set variables
ModelName=$1
Classes=$2

if [ $2 -eq 1 ]
then
    echo "Saving $ModelName parameters to ./model_parameters/binary/$ModelName.params"
    cp ./start_binary_model.com ./model_parameters/binary/$ModelName.params
fi

if [ $2 -eq 6 ]
then
    echo "Saving $ModelName parameters to ./model_parameters/multiclass/$ModelName.params"
    cp ./start_mc_model.com ./model_parameters/multiclass/$ModelName.params
fi

if [ $2 -ne 1 ] && [ $2 -ne 6 ]
then
    echo "Error. Number of classes is wrong, please try again"
fi