#!/bin/bash
#SBATCH --job-name=colorNet                 # Job name
#SBATCH --mail-type=END,FAIL         	    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rishabh.das@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --mem=16gb                          # Job memory request
#SBATCH --output=run_test_script_log%j.log  # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=tesla:2
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --account=cis6930              # Assign group
#SBATCH --qos=cis6930

pwd; hostname; date

module load python
module load cuda
module load pytorch

echo "Running plot script on a single CPU core"

rm log.txt  error_log* run_script_log.txt

#command to train the model
#python train.py ./aug_images/ -j 1 --epochs 500 -b 32 >> run_script_log.txt

#command to generate color images
python test.py >> run_script_log.txt
