DATA=voc
TASK=15-5s    # Select one of {19-1, 15-5, 15-5s}
NUM_MEM=1000 # The number of memorized features for each previous category
GPUS=0,1     # Type gpu indices

RUN_BASE=$1  # (optional) if it is set to 1, run the base stage 
SEED=$2      # (optional)
echo run-base-${RUN_BASE}, seed-${SEED}


if [ "${TASK}" = "15-5s" ]; then
  NUM_OF_STAGES=5
  LAMB=0.99
else
  NUM_OF_STAGES=1
  LAMB=0.95
fi


if [ ${RUN_BASE} -eq 1 ]; then
  if [ -n "${SEED}" ]; then # if SEED typed
    AUX="SEED ${SEED}"
  fi
  CONFIG=configs/${DATA}/${TASK}/base.yml
  CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG} --opts ${AUX}
else
  echo Skip the base stage
fi


for ((STEP=1; STEP<=NUM_OF_STAGES; STEP++))
do
  if [ ${STEP} -eq 1 ]; then
    CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml
    if [ -n "${SEED}" ]; then # if SEED typed
      FE1=Base_${SEED}_ov_${TASK}_0_last.pt
      FE2=ALIFE-S1_${SEED}_ov_${TASK}_${STEP}_last.pt
      AUX1="SEED ${SEED} MODEL.WEIGHTS ${FE1}"
      AUX2="SEED ${SEED} MODEL.WEIGHTS ${FE2}"
    fi
  else # STEP > 1
    CONFIG=configs/${DATA}/${TASK}/M${STEP}_step1.yml
    if [ -n "${SEED}" ]; then # if SEED typed
      PREV_STEP=$(expr $STEP - 1)
      FE1=ALIFE-M-S3_${SEED}_ov_${TASK}_${PREV_STEP}_last.pt
      FE2=ALIFE-M-S1_${SEED}_ov_${TASK}_${STEP}_last.pt
      AUX1="SEED ${SEED} MODEL.WEIGHTS ${FE1}"
      AUX2="SEED ${SEED} MODEL.WEIGHTS ${FE2}"
    fi
  fi

  echo
  echo ----------------------------- Start [Step 1] @ STAGE=$STEP ----------------------------- 
  CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG} --opts ${AUX1}
  echo -----------------------------  End  [Step 1] @ STAGE=$STEP ----------------------------- 
  echo



  echo
  echo -------------------- Start [Step 2: Extract features] @ STAGE=$STEP -------------------- 
  CUDA_VISIBLE_DEVICES=${GPUS} python extract_mem.py --config-file ${CONFIG} --mem-size ${NUM_MEM} --opts ${AUX1}
  echo --------------------  End  [Step 2: Extract features] @ STAGE=$STEP -------------------- 
  echo



  echo
  echo --------------------- Start [Step 2: Train matrices] @ STAGE=$STEP --------------------- 
  CUDA_VISIBLE_DEVICES=${GPUS} python train_matrices.py --config-file ${CONFIG} \
                      --mem-size ${NUM_MEM} --lamb ${LAMB}\
                      --opts SOLVER.LR 1e-2 SOLVER.MAX_EPOCH 10 ${AUX1}
  echo ---------------------  End  [Step 2: Train matrices] @ STAGE=$STEP --------------------- 
  echo



  echo
  echo --------------------- Start [Step 2: Update features] @ STAGE=$STEP -------------------- 
  CUDA_VISIBLE_DEVICES=${GPUS} python update_mem.py --config-file ${CONFIG} --mem-size ${NUM_MEM} --opts ${AUX1}
  echo ---------------------  End  [Step 2: Update features] @ STAGE=$STEP -------------------- 
  echo



  echo
  echo ----------------------------- Start [Step 3] @ STAGE=$STEP ----------------------------- 
  CONFIG=configs/${DATA}/${TASK}/M${STEP}_step3.yml
  CUDA_VISIBLE_DEVICES=${GPUS} python run_step3.py --config-file ${CONFIG} --mem-size ${NUM_MEM} --opts ${AUX2}
  echo -----------------------------  End  [Step 3] @ STAGE=$STEP ----------------------------- 
  echo
done

