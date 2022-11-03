DATA=ade
TASK=100-50  # Select one of {100-50, 50, 100-10}
GPUS=0,1     # Type gpu indices

RUN_BASE=$1  # (optional) if it is set to 1, run the base stage
SEED=$2      # (optional)
echo run-base-${RUN_BASE}, seed-${SEED}

if [ "${TASK}" = "100-10" ]; then
  NUM_OF_STAGES=5
elif [ "${TASK}" = "50" ]; then
  NUM_OF_STAGES=2
else
  NUM_OF_STAGES=1
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
  if [ -n "${SEED}" ]; then # if SEED typed
    if [ ${STEP} -eq 1 ]; then
      FE=Base_${SEED}_ov_${TASK}_0_last.pt
    else # STEP > 1
      PREV_STEP=$(expr $STEP - 1)
      FE=ALIFE-S1_${SEED}_ov_${TASK}_${PREV_STEP}_last.pt
    fi
    AUX="SEED ${SEED} MODEL.WEIGHTS ${FE}"
  fi

  echo
  echo ----------------------------- Start [Step 1] @ STAGE=$STEP ----------------------------- 
  CONFIG=configs/${DATA}/${TASK}/${STEP}_step1.yml
  CUDA_VISIBLE_DEVICES=${GPUS} python run_step1.py --config-file ${CONFIG} --opts ${AUX}
  echo -----------------------------  End  [Step 1] @ STAGE=$STEP ----------------------------- 
  echo
done

