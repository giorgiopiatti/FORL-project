sbatch --gpus=rtx_3090:1 --wrap="python3 ./ppo_com_universal.py --track --exp-name P2_PPO_universal_COM_h=32_more_samples --sample-batch-env False --hidden-dim 32 --briscola-communicate" --time=168:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16 -A ls_lawecon
sbatch --gpus=rtx_3090:1 --wrap="python3 ./ppo_com_universal.py --track --exp-name P2_PPO_universal_COM_h=64_more_samples --sample-batch-env False --hidden-dim 64 --briscola-communicate" --time=168:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16 -A ls_lawecon
sbatch --gpus=rtx_3090:1 --wrap="python3 ./ppo_com_universal.py --track --exp-name P2_PPO_universal_COM_h=32 --sample-batch-env True --hidden-dim 32 --briscola-communicate" --time=168:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16 -A ls_lawecon
sbatch --gpus=rtx_3090:1 --wrap="python3 ./ppo_com_universal.py --track --exp-name P2_PPO_universal_COM_h=64 --sample-batch-env True --hidden-dim 64 --briscola-communicate" --time=168:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16 -A ls_lawecon