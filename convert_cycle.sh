#!/bin/bash

# Sources
# ORIGINAL_FOLDERS=("/home/hlekanne/PyTorch-GAN/data/simulated2real/testA" "/home/hlekanne/PyTorch-GAN/data/simulated2real/trainA") 
# REAL_FOLDERS=("/home/hlekanne/PyTorch-GAN/data/simulated2real/testB" "/home/hlekanne/PyTorch-GAN/data/simulated2real/trainB")

# Structure
MAIN_FOLDER="/home/hlekanne/conversion_cycle/"
TRAIN="${MAIN_FOLDER}trainA"
VAL="${MAIN_FOLDER}valA"

#Scripts
CYCLE_SCRIPT_LOCATION="/home/hlekanne/PyTorch-GAN/implementations/cyclegan/"
MUNIT_SCRIPT_LOCATION="/home/hlekanne/PyTorch-GAN/implementations/munit/"

# Model location
CYCLE_MODEL_FOLDER="/home/hlekanne/PyTorch-GAN/implementations/cyclegan/saved_models/simulated2real"
MUNIT_MODEL_FOLDER="/home/hlekanne/PyTorch-GAN/implementations/munit/saved_models/simulated2real"

# Data location
TRAIN_LOCATION="/home/hlekanne/detectron2/datasets/coco/bonus/synthetic_train2017"
VAL_LOCATION="/home/hlekanne/detectron2/datasets/coco/bonus/synthetic_test2017"

# Parameters
CYCLE_GAN_PARAMETERS_TRAIN="--input_location ${TRAIN_LOCATION} --output_location ${TRAIN} --model_location ${CYCLE_MODEL_FOLDER} --model_number 202 --img_height 240 --img_width 320"
CYCLE_GAN_PARAMETERS_VAL="--input_location ${VAL_LOCATION} --output_location ${VAL} --model_location ${CYCLE_MODEL_FOLDER} --model_number 202 --img_height 240 --img_width 320"

rm ${MAIN_FOLDER} -r

mkdir ${MAIN_FOLDER}
mkdir ${TRAIN}
mkdir ${VAL}

eval "$(conda shell.bash hook)"
conda activate stargan-v2

cd ${CYCLE_SCRIPT_LOCATION}
python3 cyclegan_input.py ${CYCLE_GAN_PARAMETERS_TRAIN}
python3 cyclegan_input.py ${CYCLE_GAN_PARAMETERS_VAL}

conda deactivate
conda activate PerceptualSimilarity

