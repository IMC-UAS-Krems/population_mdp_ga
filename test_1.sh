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
  