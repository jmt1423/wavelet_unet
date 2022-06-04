#!/bin/bash

rm ./binary-model.e*
rm ./binary-model.o*
rm ./multi-class-model.e*
rm ./multi-class-model.o*
rm ./hyper-parameter-optim.e*
rm ./hyper-parameter-optim.o*