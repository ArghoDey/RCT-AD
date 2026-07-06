#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28651}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --no-validate \
    --work-dir work_dirs/your_path
