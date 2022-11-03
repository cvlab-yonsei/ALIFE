DATA=ade
TASK=100-50  # Select one of {100-50, 50, 100-10}
GPUS=0,1     # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/base.yml
CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG}
