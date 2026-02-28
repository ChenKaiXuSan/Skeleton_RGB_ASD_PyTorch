#!/bin/bash
#PBS -A SKIING
#PBS -q gen_S
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_person_20_only
#PBS -o logs/pegasus/sam3d_p20.log
#PBS -e logs/pegasus/sam3d_p20_err.log

# === 1. 環境準備 ===
cd /work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch
mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

# === 2. パス設定 ===
VIDEO_PATH="/work/SKIING/chenkaixu/data/drive/videos_split"
RESULT_PATH="/work/SKIING/chenkaixu/data/drive/sam3d_body_results_2"
CKPT_ROOT="/work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch/ckpt/sam-3d-body-dinov3"

echo "🚀 Starting SAM3Dbody inference"
echo "Started at: $(date)"

# === 4. 実行 ===
# 扁平视频目录推理
python -m SAM3Dbody.main \
    paths.video_path=${VIDEO_PATH} \
    paths.result_output_path=${RESULT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=7

echo "✅ Finished Person 20 at: $(date)"