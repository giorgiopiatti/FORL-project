
# David 
# Initial Euler setup
- Ensure python 3.9 is used
pip install -r requirements.txt
mkdir /cluster/scratch/dgu/FORL_code
ln -s /cluster/scratch/dgu/FORL_code ./FORL_Briscola

# Load file auf Euler
rsync --exclude 'log' --exclude 'wandb'  -r ./  dgu@euler.ethz.ch:FORL_Briscola


# Submit job 
sbatch --gpus=1  --wrap="python caller_callee_heuristic_DQN.py --logger wandb --logdir /cluster/scratch/dgu/FORL_briscola/ " --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A s_stud_infk


# Nat 
# Initial Euler setup
- Ensure python 3.9 is used
pip install -r requirements.txt
mkdir /cluster/scratch/ncorecco/FORL_code
ln -s /cluster/scratch/ncorecco/FORL_code ./FORL_Briscola

# Load file auf Euler
rsync --exclude 'log' --exclude 'wandb'  -r ./  ncorecco@euler.ethz.ch:FORL_Briscola


# Submit job 
sbatch --gpus=1  --wrap="python caller_callee_heuristic_DQN.py --logger wandb --logdir /cluster/scratch/ncorecco/FORL_briscola/ " --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A s_stud_infk
