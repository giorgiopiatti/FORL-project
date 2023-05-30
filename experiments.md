RRN NOCOM
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=32_rnn=64 --sample-batch-env False --hidden-dim 32 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=32_rnn=128 --sample-batch-env False --hidden-dim 32 --rnn-out-size 128

python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=64_rnn=64 --sample-batch-env False --hidden-dim 64 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=64_rnn=128 --sample-batch-env False --hidden-dim 64 --rnn-out-size 128

python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=128_rnn=64 --sample-batch-env False --hidden-dim 128 --rnn-out-size 64
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_RNN_h=128_rnn=128 --sample-batch-env False --hidden-dim 128 --rnn-out-size 128


NN NOCOM
python3 ./ppo_com_universal.py --track --exp-name PPO_universal_h=32_more_samples --sample-batch-env False --hidden-dim 32
python3 ./ppo_com_universal.py --track --exp-name PPO_universal_h=64_more_samples --sample-batch-env False --hidden-dim 64
python3 ./ppo_com_universal.py --track --exp-name PPO_universal_h=32 --sample-batch-env True --hidden-dim 32
python3 ./ppo_com_universal.py --track --exp-name PPO_universal_h=64 --sample-batch-env True --hidden-dim 64


sbatch --gpus=rtx_2080_ti:1 --wrap="" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon

sbatch --gpus=1 --gres=gpumem:20g --wrap="p" --time=120:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon


RNN & COM:
RRN NOCOM
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=32_rnn=64 --sample-batch-env False --hidden-dim 32 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=32_rnn=128 --sample-batch-env False --hidden-dim 32 --rnn-out-size 128 --briscola-communicate

python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=64_rnn=64 --sample-batch-env False --hidden-dim 64 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=64_rnn=128 --sample-batch-env False --hidden-dim 64 --rnn-out-size 128 --briscola-communicate

python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=128_rnn=64 --sample-batch-env False --hidden-dim 128 --rnn-out-size 64 --briscola-communicate
python3 ./ppo_RNN_universal.py --track --exp-name PPO_universal_COM_RNN_h=128_rnn=128 --sample-batch-env False --hidden-dim 128 --rnn-out-size 128 --briscola-communicate