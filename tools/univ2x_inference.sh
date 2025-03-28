export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# 运行 Python 推理脚本
python tools/inference.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_old_mode_inference_wo_label.py \
    ckpts/univ2x_coop_e2e_stg2_old_mode_inference_only.pth \
    --out output/results_to_submit.pkl \
    --launcher pytorch
