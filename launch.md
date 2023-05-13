python3 ./ppo.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple
python3 ./ppo.py --briscola-train-mode  bad_single_network --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_single

python3 ./ppo.py --briscola-roles-train caller --briscola-callee-heuristic --track --exp-name PPO_caller_heuristic_callee
python3 ./ppo.py --briscola-roles-train callee --briscola-caller-heuristic --track --exp-name PPO_callee_heuristic_calleer

python3 ./ppo.py --briscola-roles-train caller --track --exp-name PPO_caller
python3 ./ppo.py --briscola-roles-train callee --track --exp-name PPO_callee

sbatch --gpus=rtx_2080_ti:1 --wrap="python3 ./ppo.py --briscola-train-mode  bad_multiple_networks --num-generations 5 --total-timesteps 8000000 --track --exp-name PPO_bad_multiple" --time=48:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A ls_lawecon




--track --freq-eval-test 100 --briscola-train
-mode bad_multiple_networks --total-timesteps 800 --freq-eval-test 1 --num-test-games 1 