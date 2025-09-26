#!/bin/bash

COMMAND="python main.py --idi_num 100 --log"
COMMAND="python main.py --idi_num 100 --device cuda:0"
DATA=$1
MODE="nor1+nor2"

if [ "$DATA" = "compas" ]; then
  SENS=("sex" "race" "age")
  MODEL="compas1"
  datanum=500
elif [ "$DATA" = "german" ]; then
  SENS=("sex" "age")
  MODEL="GC1"
  datanum=100
  EPSPHI=("G1")
elif [ "$DATA" = "adult" ]; then
  SENS=("sex" "age" "race")
  MODEL="AC1"
  datanum=500
  EPSPHI=("A1")
elif [ "$DATA" = "bank" ]; then
  SENS=("age" )
  MODEL="BM0"
  datanum=500
  EPSPHI=("B1")
else
  echo "Unsupported dataset: $DATA"
  exit 1
fi

for SEN in "${SENS[@]}"; do
  for EPS in "${EPSPHI[@]}"; do
    FULL_COMMAND="$COMMAND --dataset $DATA --model $MODEL --SA $SEN --data_mode $MODE --data_num $datanum --eps_prop $EPS"
    echo "Running: $FULL_COMMAND"
    $FULL_COMMAND
  done
done

