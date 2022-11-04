DATA=voc
TASK=15-5 # Select one of {19-1, 15-5, 15-5s} 
STEP=1    # Type a valid step (> 0)
GPUS=0,1  # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml # For ALIFE
#CONFIG=configs/${DATA}/${TASK}/M${STEP}_step1.yml # For ALIFE-M
CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG}
