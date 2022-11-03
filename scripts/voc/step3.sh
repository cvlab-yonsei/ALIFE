DATA=voc
TASK=15-5    # Select one of {19-1, 15-5, 15-5s}
STEP=1       # Type a valid step (> 0)
NUM_MEM=1000 # The number of memorized features for each previous category
GPUS=0,1     # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/M${STEP}_step3.yml
CUDA_VISIBLE_DEVICES=${GPUS} python run_step3.py --config-file ${CONFIG} --mem-size ${NUM_MEM}

