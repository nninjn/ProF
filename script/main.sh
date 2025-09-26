#!/bin/bash

COMMAND="python main.py --idi_num 100"
DATA=$1

if [ "$DATA" = "compas" ]; then
  SENS=("sex" "race" "age")
  MODEL="compas1"
  datanum=500
elif [ "$DATA" = "german" ]; then
  SENS=("sex" "age")
  MODEL="GC1"
  datanum=100
elif [ "$DATA" = "adult" ]; then
  SENS=("sex" "age" "race")
  MODEL="AC1"
  datanum=500
elif [ "$DATA" = "bank" ]; then
  SENS=("age" )
  MODEL="BM0"
  datanum=500
else
  echo "Unsupported dataset: $DATA"
  exit 1
fi

for SEN in "${SENS[@]}"; do
    FULL_COMMAND="$COMMAND --dataset $DATA --model $MODEL --SA $SEN --data_num $datanum"
    echo "Running: $FULL_COMMAND"
    $FULL_COMMAND
done

