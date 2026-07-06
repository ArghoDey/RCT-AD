## RCT-AD evaluation
## Stage 1
bash ./tools/dist_test.sh \
    projects/configs/RCT-AD_stage1.py \
    work_dirs/your_path.pth \
    8 \
    --deterministic \
    --eval bbox

## Stage 2
bash ./tools/dist_test.sh \
    projects/configs/RCT-AD_stage2.py \
    work_dirs/your_path.pth \
    8 \
    --deterministic \
    --eval bbox
