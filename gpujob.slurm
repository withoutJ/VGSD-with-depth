#!/bin/bash
#SBATCH --job-name=train_VGSD_with_depth
#SBATCH --partition=gpu_v100s
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=55GB               # Request 50GB memory
#SBATCH --time=0-02:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=gpujob_%j.log   # Standard output
#SBATCH --error=gpujob_%j.err    # Standard error log
#SBATCH --mail-user=mmrkaic2-c@my.cityu.edu.hk

OUTPUT=/home/mmrkaic/scratch/train_VGSD_with_depth.out
cd /home/mmrkaic/scratch/VGSD_with_depth

python train.py --max_epoch 10 --batch_size 4 --num_workers 4 >> $OUTPUT