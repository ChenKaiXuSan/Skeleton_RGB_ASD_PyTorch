#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_4nodes_run
#PBS -t 0-21                          # 22个
#PBS -o logs/pegasus/sam3d_group_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_group_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
cd /work/SSR/share/code/Drive_Face_Mesh_PyTorch

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

echo "Node Index: $PBS_SUBREQNO"

# === 3. パス設定と実行 ===
VIDEO_PATH="/work/SSR/share/data/drive/videos_split"
RESULT_PATH="/work/SSR/share/data/drive/sam3d_body_results_right"
CKPT_ROOT="/work/SSR/share/code/Drive_Face_Mesh_PyTorch/ckpt/sam-3d-body-dinov3"

echo "🏁 Node ${PBS_SUBREQNO} started at: $(date)"

python -m SAM3Dbody.main \
    paths.video_path=${VIDEO_PATH} \
    paths.result_output_path=${RESULT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=7

echo "🏁 Node ${PBS_SUBREQNO} finished at: $(date)"
# 一个node里面跑一个人的4个环境，也就是4个worker