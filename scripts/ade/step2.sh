DATA=ade
TASK=100-50  # Select one of {100-50, 50, 100-10}
STEP=1       # Type a valid step (> 0)
NUM_MEM=1000 # The number of memorized features for each previous category
GPUS=0,1     # Type gpu indices

if [ ${STEP} -eq 1 ]; then
  CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml
else # STEP > 1
  CONFIG=configs/${DATA}/${TASK}/M${STEP}_step1.yml
fi

# --------------------------------
#        Extract features
# --------------------------------
CUDA_VISIBLE_DEVICES=${GPUS} python extract_mem.py --config-file ${CONFIG} --mem-size ${NUM_MEM}
# --------------------------------



# --------------------------------
#    Train rotation matrices
# --------------------------------
CUDA_VISIBLE_DEVICES=${GPU} python train_matrices.py --config-file ${CONFIG} \
                    --mem-size ${NUM_MEM} --opts SOLVER.LR 1e-2 SOLVER.MAX_EPOCH 10
# --------------------------------



# --------------------------------
#        Update features
# --------------------------------
CUDA_VISIBLE_DEVICES=${GPUS} python update_mem.py --config-file ${CONFIG} --mem-size ${NUM_MEM}
# --------------------------------

