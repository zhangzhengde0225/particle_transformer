#! /bin/bash

#======== part 1 : job parameters =========
#SBATCH --partition=spub
#SBATCH --account=u07
#SBATCH --qos=spubregular
#SBATCH --mem-per-cpu=4GB
#SBATCH --ntasks=2
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=triain_ai

#======== part 2 : workload ========
echo "slurm job ${SLURM_JOB_ID} start.."
date

# job worklaod comes here
source /cvmfs/hai.ihep.ac.cn/hai_env.sh
# add your programs 
python train.py \
    --name particle_transformer \
    --source JetClass-mini \
    --feature-type full \
    --device 0 

echo "slurm job ${SLURM_JOB_ID} done."
date
