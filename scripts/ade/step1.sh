DATA=ade
TASK=100-50 # Select one of {100-50, 50, 100-10}
STEP=1      # Type a valid step (> 0)
GPUS=0,1    # Type gpu indices

if [ ${STEP} -eq 1 ]; then
  CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml
else # STEP > 1
  CONFIG=configs/${DATA}/${TASK}/M${STEP}_step1.yml
fi
CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG}
