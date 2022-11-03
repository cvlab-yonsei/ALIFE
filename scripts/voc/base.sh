DATA=voc
TASK=15-5 # Select one of {19-1, 15-5, 15-5s}
GPUS=0,1  # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/base.yml
CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG}
