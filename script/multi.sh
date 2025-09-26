#!/bin/bash

COMMAND="python main.py --idi_num 100 --log"
COMMAND="python main.py --idi_num 100 --device cuda:0"
DATA=$1
MODES=("nor1+nor2")

if [ "$DATA" = "compas" ]; then
  SENS=("sex_race" "sex_age" "race_age")
  MODEL="compas1"
  datanum=500
elif [ "$DATA" = "german" ]; then
  SENS=("sex_age")
  MODEL="GC1"
  datanum=100
elif [ "$DATA" = "adult" ]; then
  SENS=("sex_race" "sex_age" "race_age")
  MODEL="AC1"
  datanum=500
else
  echo "Unsupported dataset: $DATA"
  exit 1
fi

for SEN in "${SENS[@]}"; do
  for MODE in "${MODES[@]}"; do
    FULL_COMMAND="$COMMAND --dataset $DATA --model $MODEL --SA $SEN --data_mode $MODE --data_num $datanum"
    echo "Running: $FULL_COMMAND"
    $FULL_COMMAND
  done
done

