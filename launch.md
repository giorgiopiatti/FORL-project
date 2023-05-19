python3 ./ppo.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple_points
python3 ./ppo.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single_points

python3 ./ppo.py --briscola-roles-train caller --briscola-callee-heuristic --track --exp-name PPO_caller_heuristic_callee
python3 ./ppo.py --briscola-roles-train callee --briscola-caller-heuristic --track --exp-name PPO_callee_heuristic_calleer

python3 ./ppo.py --briscola-roles-train caller --track --exp-name PPO_caller
python3 ./ppo.py --briscola-roles-train callee --track --exp-name PPO_callee





python3 ./ppo_com.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single_COMS --briscola-communicate
python3 ./ppo_com.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single_COMS_truth --briscola-communicate --briscola-communicate-truth-only


python3 ./ppo_com.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple_COMS --briscola-communicate

python3 ./ppo_com.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple_COMS_truth --briscola-communicate --briscola-communicate-truth-only

sbatch --gpus=rtx_2080_ti:1 --wrap="" --time=48:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon




--track --freq-eval-test 100 --briscola-train
-mode bad_multiple_networks --total-timesteps 800 --freq-eval-test 1 --num-test-games 1 




IMMEDIATE REWARD
--briscola-env-immediate-reward

python3 ./ppo.py --briscola-roles-train caller --track --exp-name PPO_caller_IMMEDIATE_REWARD
python3 ./ppo.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple_points_IMMEDIATE_REWARD
python3 ./ppo.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single_points_IMMEDIATE_REWARD


sbatch --gpus=rtx_2080_ti:1 --wrap="python3 ./ppo.py --briscola-roles-train caller --track --exp-name PPO_caller_IMMEDIATE_REWARD" --time=48:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon

sbatch --gpus=rtx_2080_ti:1 --wrap="python3 ./ppo.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple_points_IMMEDIATE_REWARD" --time=48:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon

sbatch --gpus=rtx_2080_ti:1 --wrap="python3 ./ppo.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single_points_IMMEDIATE_REWARD" --time=48:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon