RRN NOCOM
python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_RNN_h=32_rnn=64 --sample-batch-env False --hidden-dim 32 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_RNN_h=32_rnn=128 --sample-batch-env False --hidden-dim 32 --rnn-out-size 128

python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_RNN_h=64_rnn=64 --sample-batch-env False --hidden-dim 64 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_RNN_h=64_rnn=128 --sample-batch-env False --hidden-dim 64 --rnn-out-size 128

python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_RNN_h=128_rnn=64 --sample-batch-env False --hidden-dim 128 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --cuda False  --exp-name P2_PPO_universal_RNN_h=128_rnn=128 --sample-batch-env False --hidden-dim 128 --rnn-out-size 128 
NEED TO BE RUNNED for old probs


NN NOCOM
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_h=32_more_samples --sample-batch-env False --hidden-dim 32
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_h=64_more_samples --sample-batch-env False --hidden-dim 64
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_h=32 --sample-batch-env True --hidden-dim 32
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_h=64 --sample-batch-env True --hidden-dim 64


NN COM
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_COM_h=32_more_samples --sample-batch-env False --hidden-dim 32 --briscola-communicate
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_COM_h=64_more_samples --sample-batch-env False --hidden-dim 64 --briscola-communicate
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_COM_h=32 --sample-batch-env True --hidden-dim 32 --briscola-communicate
python3 ./ppo_com_universal.py --cuda False --track --exp-name P2_PPO_universal_COM_h=64 --sample-batch-env True --hidden-dim 64 --briscola-communicate



sbatch --gpus=rtx_2080_ti:1 --wrap="" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon

sbatch --gpus=1 --gres=gpumem:20g --wrap="p" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon
sbatch -C fabric8 --wrap="" --time=168:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 -A ls_lawecon


RNN & COM:
python3 ./ppo_RNN_universal.py --track  --cuda False --exp-name P2_PPO_universal_COM_RNN_h=32_rnn=64 --sample-batch-env False --hidden-dim 32 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track  --cuda False --exp-name P2_PPO_universal_COM_RNN_h=32_rnn=128 --sample-batch-env False --hidden-dim 32 --rnn-out-size 128 --briscola-communicate

python3 ./ppo_RNN_universal.py --track  --cuda False --exp-name P2_PPO_universal_COM_RNN_h=64_rnn=64 --sample-batch-env False --hidden-dim 64 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track  --cuda False --exp-name P2_PPO_universal_COM_RNN_h=64_rnn=128 --sample-batch-env False --hidden-dim 64 --rnn-out-size 128 --briscola-communicate

python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_COM_RNN_h=128_rnn=64 --sample-batch-env False --hidden-dim 128 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track --cuda False --exp-name P2_PPO_universal_COM_RNN_h=128_rnn=128 --sample-batch-env False --hidden-dim 128 --rnn-out-size 128 --briscola-communicate


sbatch --gpus=rtx_2080_ti:1 --wrap="" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon

sbatch --gpus=1 --gres=gpumem:20g --wrap="python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=32_rnn=64                          --sample-batch-env False --hidden-dim 32 --rnn-out-size 64 --briscola-communicate --resume-path ./checkpoints/pretrained/j7h7xjb5_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon


sbatch -C fabric8 --wrap="" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16 -A ls_lawecon



2ND phase
python3 ./ppo_com_universal.py --track  --cuda False --exp-name 2PHASE_PPO_universal_COM_h=64_more_samples                          --sample-batch-env False --hidden-dim 64 --briscola-communicate --resume-path ./checkpoints/pretrained/1mfxc6le_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_com_universal.py --track  --cuda False --exp-name 2PHASE_PPO_universal_COM_h=32_more_samples                          --sample-batch-env False --hidden-dim 32 --briscola-communicate --resume-path ./checkpoints/pretrained/2butggxi_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_com_universal.py --track  --cuda False --exp-name 2PHASE_PPO_universal_COM_h=32                          --sample-batch-env True --hidden-dim 32 --briscola-communicate --resume-path ./checkpoints/pretrained/2he5v1mm_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_com_universal.py --track  --cuda False --exp-name 2PHASE_PPO_universal_COM_h=64                          --sample-batch-env True --hidden-dim 64 --briscola-communicate --resume-path ./checkpoints/pretrained/cziybs7v_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000


sbatch --gpus=rtx_2080_ti:1 --wrap="" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon


python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=128_rnn=64                          --sample-batch-env False --hidden-dim 128 --rnn-out-size 64 --briscola-communicate --resume-path ./checkpoints/pretrained/wzvf77xg_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=64_rnn=128                          --sample-batch-env False --hidden-dim 64 --rnn-out-size 128 --briscola-communicate --resume-path ./checkpoints/pretrained/euihftzr_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=64_rnn=64                          --sample-batch-env False --hidden-dim 64 --rnn-out-size 64 --briscola-communicate --resume-path ./checkpoints/pretrained/dicdr57l_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=32_rnn=128                          --sample-batch-env False --hidden-dim 32 --rnn-out-size 128 --briscola-communicate --resume-path ./checkpoints/pretrained/92aoa9l2_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

python3 ./ppo_RNN_universal.py --track  --cuda True --exp-name 2PHASE_PPO_universal_COM_RNN_h=32_rnn=64                          --sample-batch-env False --hidden-dim 32 --rnn-out-size 64 --briscola-communicate --resume-path ./checkpoints/pretrained/j7h7xjb5_base.pt                             --briscola-communicate-truth False --total-timesteps 96000000 --freq-save-model 10000

