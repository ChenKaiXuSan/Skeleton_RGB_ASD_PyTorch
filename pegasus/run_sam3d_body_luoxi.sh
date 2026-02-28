#!/bin/bash
#PBS -N sam3d_by_disease
#PBS -t 0-3
#PBS -o logs/pegasus/sam3d_disease_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_disease_${PBS_SUBREQNO}_err.log
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00

# === 1. 環境準備 ===
cd /work/SSR/share/code/Skeleton_RGB_ASD_PyTorch

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

echo "Job ID: ${PBS_JOBID}"
echo "Sub Job Index: ${PBS_SUBREQNO}"

# === 2. 疾病マッピング（1サブジョブ=1疾病） ===
DISEASES=("ASD" "DHS" "LCS" "HipOA")
DISEASE=${DISEASES[$PBS_SUBREQNO]}

if [ -z "${DISEASE}" ]; then
    echo "❌ Invalid PBS_SUBREQNO=${PBS_SUBREQNO}, expected 0-3"
    exit 1
fi

echo "Disease: ${DISEASE}"

# === 3. パス設定と実行 ===
VIDEO_PATH="/work/SSR/share/data/asd_dataset/skeleton_rgb_dataset/video"
RESULT_PATH="/work/SSR/share/data/asd_dataset/skeleton_rgb_dataset/sam3d_body_results"
CKPT_ROOT="/work/SSR/share/code/Drive_Face_Mesh_PyTorch/ckpt/sam-3d-body-dinov3"

echo "🏁 Job started at: $(date)"

python -m SAM3Dbody.main \
    paths.video_path=${VIDEO_PATH} \
    paths.result_output_path=${RESULT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=6 \
    infer.disease_list="[${DISEASE}]"

echo "🏁 Job finished at: $(date)"