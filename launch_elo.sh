max=7
for i in `seq 0 $max`
do
    sbatch --wrap="python3 ./ELO_parallel.py --briscola-communicate --briscola-communicate-truth False --split $i --path-pool ./ELO_tournament/com --file-pool ./ELO_tournament/com_models_new.csv --path-scores ./ELO_tournament/com_scores_lies_split.csv --num-workers 128" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=128 -A ls_lawecon
done