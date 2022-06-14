#! /bin/bash

NUM = $1
mkdir -p /storage/hpc/27/thomann/coastal_segmentation_data/current_data/train{$NUM}/

for i in $(seq 1 $NUM)
    mv 