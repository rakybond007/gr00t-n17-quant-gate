#!/bin/bash
#SBATCH --job-name=libero_tile_generation_stride1_full_frame_cpu_only_no_gpu_needed_decord_decode
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --array=0-15
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# **GPU 를 안 쓴다.** gen_libero_tiles_shard.py 는 decord 로 영상을 디코드하고 PIL 로
# 타일을 굽는다 -- torch 도 cuda 도 안 부른다. background 에서 GPU 를 물고 대기하는
# 것은 낭비이고, 라벨링 잡과 같은 파티션을 두고 경쟁하게 된다.
# 이미 있는 타일은 다시 그리지 않으므로 background 판과 같이 돌아도 협력한다.
BASE=$HOME/quantization_agent_workspace; cd $BASE
NSH=$((SLURM_ARRAY_TASK_MAX+1))
TILE_STRIDE=${TILE_STRIDE:-1} $HOME/miniconda3/envs/quant_gate/bin/python -u \
  vlm_gate/scripts/gen_libero_tiles_shard.py $SLURM_ARRAY_TASK_ID $NSH
