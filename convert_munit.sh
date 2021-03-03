#!/bin/bash

# Sources
# ORIGINAL_FOLDERS=("/home/hiddelekanne/PyTorch-GAN/data/simulated2real/testA" "/home/hiddelekanne/PyTorch-GAN/data/simulated2real/trainA") 
# REAL_FOLDERS=("/home/hiddelekanne/PyTorch-GAN/data/simulated2real/testB" "/home/hiddelekanne/PyTorch-GAN/data/simulated2real/trainB")

# Structure
MAIN_FOLDER="/home/hiddelekanne/conversion_munit/"
TRAIN="${MAIN_FOLDER}train2017"
VAL="${MAIN_FOLDER}val2017"

#Scripts
CYCLE_SCRIPT_LOCATION="/home/hiddelekanne/PyTorch-GAN/implementations/cyclegan/"
MUNIT_SCRIPT_LOCATION="/home/hiddelekanne/PyTorch-GAN/implementations/munit/"

# Model location
CYCLE_MODEL_FOLDER="/home/hiddelekanne/PyTorch-GAN/implementations/cyclegan/saved_models/simulated2real"
MUNIT_MODEL_FOLDER="/home/hiddelekanne/PyTorch-GAN/implementations/munit/saved_models/simulated2real"

# Data location
TRAIN_LOCATION="/home/hiddelekanne/detectron2/datasets/coco/Versions/Pure_Simulation/train2017"
VAL_LOCATION="/home/hiddelekanne/detectron2/datasets/coco/Versions/Pure_Simulation/val2017"

# Parameters
MUNIT_GAN_PARAMETERS_TRAIN="--input_location ${TRAIN_LOCATION} --output_location ${TRAIN} --model_location ${MUNIT_MODEL_FOLDER} --model_number 29 --img_height 240 --img_width 320"
MUNIT_GAN_PARAMETERS_VAL="--input_location ${VAL_LOCATION} --output_location ${VAL} --model_location ${MUNIT_MODEL_FOLDER} --model_number 29 --img_height 240 --img_width 320"

rm ${MAIN_FOLDER} -r

mkdir ${MAIN_FOLDER}
mkdir ${TRAIN}
mkdir ${VAL}

eval "$(conda shell.bash hook)"
conda activate stargan-v2

cd ${MUNIT_SCRIPT_LOCATION}
python3 munit_input.py ${MUNIT_GAN_PARAMETERS_TRAIN}

cd ${MUNIT_SCRIPT_LOCATION}
python3 munit_input.py ${MUNIT_GAN_PARAMETERS_VAL}

conda deactivate
conda activate PerceptualSimilarity

