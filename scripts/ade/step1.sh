DATA=ade
TASK=100-50 # Select one of {100-50, 50, 100-10}
STEP=1      # Type a valid step (> 0)
GPUS=0,1    # Type gpu indices

CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml # For ALIFE
#CONFIG=configs/${DATA}/${TASK}/M${STEP}_step1.yml # For ALIFE-M
CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG}
