## RCT-AD training
## Stage 1: perception-only pre-training
bash ./tools/dist_train.sh \
   projects/configs/RCT-AD_stage1.py \
   8 \
   --deterministic

## Stage 2: unified end-to-end optimization
bash ./tools/dist_train.sh \
   projects/configs/RCT-AD_stage2.py \
   4 \
   --deterministic
