#!/bin/bash
#SBATCH --account=bonvinc
#SBATCH --partition=shared-gpu
#SBATCH --job-name=alpaca
#SBATCH --output=logs/alpaca_%j.out
#SBATCH --error=logs/alpaca_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hrvoje.krizic@unige.ch

# --- 1. Load Environment ---
#module load python/3.9
#source /home/users/k/krizich/miniforge/etc/profile.d/conda.sh
#conda activate herculens-env

# --- 2. Move to Directory ---
#cd $HOME/code/Alpaca_Try

# --- 3. Stability Flags ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=eth,enp,lo

# --- 4. JAX Flags ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export NVIDIA_TF32_OVERRIDE=0

# --- 5. Run ---
python -u run_alpaca.py
