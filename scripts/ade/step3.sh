DATA=ade
TASK=100-50  # Select one of {100-50, 50, 100-10}
STEP=1       # Type a valid step (> 0)
NUM_MEM=1000 # The number of memorized features for each previous category
GPUS=0,1     # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/M${STEP}_step3.yml
CUDA_VISIBLE_DEVICES=${GPUS} python run_step3.py --config-file ${CONFIG} --mem-size ${NUM_MEM}

