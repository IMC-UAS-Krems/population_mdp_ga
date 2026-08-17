python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem onemax \
  --crossover-rate 0.9 \
  --output-prefix ga_mdp_onemax

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem onemax \
  --crossover-rate 0.0 \
  --output-prefix ga_mdp_onemax_mutation_only

python plot_combined_ga_results.py \
    ga_mdp_onemax_mutation_only.csv \
    ga_mdp_onemax.csv \
    --labels \
        'Mutation only ($p_c=0$)' \
        'Crossover ($p_c=0.9$)' \
    --output-prefix onemax_combined \
    --title "OneMax: exact population-MDP predictions versus simulation"
