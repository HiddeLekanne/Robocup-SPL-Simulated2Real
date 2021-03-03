#!/bin/bash
# Sources
DATA_LOCATION="/home/hiddelekanne/detectron2/datasets/coco/Versions"
REAL=("/home/hiddelekanne/PyTorch-GAN/data/simulated2real/trainB" "/home/hiddelekanne/PyTorch-GAN/data/simulated2real/testB" "REAL")
SIMULATION_PURE=("${DATA_LOCATION}/Pure_Simulation/train2017" "${DATA_LOCATION}/Pure_Simulation/val2017" "SIMULATION_PURE")
MUNIT_NORMAL=("${DATA_LOCATION}/Munit_NORMAL_LOSS/train2017" "${DATA_LOCATION}/Munit_NORMAL_LOSS/val2017" "MUNIT_NORMAL")
MUNIT_ADJUSTED=("${DATA_LOCATION}/Munit_ADJUSTED_LOSS/train2017" "${DATA_LOCATION}/Munit_ADJUSTED_LOSS/val2017" "MUNIT_ADJUSTED") 
CYCLEGAN_NORMAL=("${DATA_LOCATION}/CycleGan_NORMAL_LOSS/train2017" "${DATA_LOCATION}/CycleGan_NORMAL_LOSS/val2017" "CYCLEGAN_NORMAL")
CYCLEGAN_ADJUSTED=("${DATA_LOCATION}/CycleGAN_ADJUSTED_LOSS/train2017" "${DATA_LOCATION}/CycleGAN_ADJUSTED_LOSS/val2017" "CYCLEGAN_ADJUSTED")

ALL_ARCHITECTURES_TRAIN=(${MUNIT_ADJUSTED[0]} ${MUNIT_NORMAL[0]} ${CYCLEGAN_NORMAL[0]} ${CYCLEGAN_ADJUSTED[0]})
ALL_ARCHITECTURES_VAL=(${MUNIT_ADJUSTED[1]} ${MUNIT_NORMAL[1]} ${CYCLEGAN_NORMAL[1]} ${CYCLEGAN_ADJUSTED[1]})
ALL_ARCHITECTURES_NAMES=(${MUNIT_ADJUSTED[2]} ${MUNIT_NORMAL[2]} ${CYCLEGAN_NORMAL[2]} ${CYCLEGAN_ADJUSTED[2]})
ALL_DATASETS_TRAIN=(${SIMULATION_PURE[0]} ${REAL[0]})
ALL_DATASETS_VAL=(${SIMULATION_PURE[1]} ${REAL[1]})
ALL_DATASETS_NAMES=(${SIMULATION_PURE[2]} ${REAL[2]})

# Structure
RANDOM_SAMPLE_SIZE=100000
MAIN_FOLDER="/home/hiddelekanne/metrics/"
RESULTS_FOLDER="${MAIN_FOLDER}"
FINAL_RESULTS_TRAIN="${MAIN_FOLDER}/all_results_train.txt"
FINAL_RESULTS_VAL="${MAIN_FOLDER}/all_results_val.txt"

#Scripts
PERCEPTUALSIMILARITY_FOLDER="/home/hiddelekanne/PerceptualSimilarity"
FID_FOLDER="/home/hiddelekanne/pytorch-fid"

rm ${MAIN_FOLDER} -r
mkdir ${MAIN_FOLDER}
# mkdir ${RESULTS_FOLDER}

eval "$(conda shell.bash hook)"
conda activate stargan-v2

cd $MAIN_FOLDER

# cd ${PERCEPTUALSIMILARITY_FOLDER}
# python3 compute_dists_internal.py -d0 ${CONVERTED_FOLDER} -o ${RESULTS_FOLDER}/cyclegan.txt --use_gpu
# python3 compute_dists_internal.py -d0 ${SYNTHETIC_SAMPLE_FOLDER} -o ${RESULTS_FOLDER}/synthetic.txt --use_gpu
# python3 compute_dists_internal.py -d0 ${REAL_SAMPLE_FOLDER} -o ${RESULTS_FOLDER}/real.txt --use_gpu

BASH_SIZE=$((50 % RANDOM_SAMPLE_SIZE))
cd ${FID_FOLDER}

i=0
for architecture in ${ALL_ARCHITECTURES_TRAIN[@]}
do 
	j=0
	for dataset in ${ALL_DATASETS_TRAIN[@]}
	do
		echo ${ALL_ARCHITECTURES_NAMES[${i}]}
		echo ${ALL_DATASETS_NAMES[${j}]}
		output_name="${MAIN_FOLDER}/${ALL_ARCHITECTURES_NAMES[${i}]}_${ALL_DATASETS_NAMES[${j}]}.txt"
		python3 fid_score.py ${architecture} ${dataset} --batch-size ${BASH_SIZE} --out "${output_name}" --gpu 0
		echo "${output_name}" >> ${FINAL_RESULTS_TRAIN}
		cat ${output_name} >> ${FINAL_RESULTS_TRAIN}
		rm ${output_name}
		j=$((${j} + 1))
	done
	i=$((${i} + 1))
done


i=0
for architecture in ${ALL_ARCHITECTURES_VAL[@]}
do 
	j=0
	for dataset in ${ALL_DATASETS_VAL[@]}
	do
		echo ${ALL_ARCHITECTURES_NAMES[${i}]}
		echo ${ALL_DATASETS_NAMES[${j}]}
		output_name="${MAIN_FOLDER}/${ALL_ARCHITECTURES_NAMES[${i}]}_${ALL_DATASETS_NAMES[${j}]}.txt"
		python3 fid_score.py ${architecture} ${dataset} --batch-size ${BASH_SIZE} --out "${output_name}" --gpu 0
		echo "${output_name}" >> ${FINAL_RESULTS_VAL}
		cat ${output_name} >> ${FINAL_RESULTS_VAL}
		rm ${output_name}
		j=$((${j} + 1))
	done
	i=$((${i} + 1))
done
