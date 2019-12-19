#!/bin/bash
echo "Usage: ./convert.sh keras_modelxxx.h5"
name=`echo $1 | cut -d '.' -f 1`
tflite_out=$name.tflite
kmodel_out=$name.kmodel

#Adjust this
ncc_exe=../../../tools/nncase-1v5/ncc
tflite_convert_exe=tflite_convert #This should be available when you install Tensorflow

echo ">> Converting H5 to TFlite"
$tflite_convert_exe --keras_model_file $1 --output_file $tflite_out

echo ">> Converting TFlite to Kmodel"
$ncc_exe -i tflite -o k210model --dataset dataset/test/BB8 $tflite_out ./$kmodel_out

echo ">> OK. Not if all goes well, copy $kmodel_out to SD card"
