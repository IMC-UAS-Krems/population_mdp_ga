# Performance Prediction Comparison for OneMax Problem

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

# Performance Prediction Comparison for LeadingOnes Problem

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem leadingones \
  --crossover-rate 0.9 \
  --output-prefix ga_mdp_leadingones

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem leadingones \
  --crossover-rate 0.0 \
  --output-prefix ga_mdp_leadingones_mutation_only

python plot_combined_ga_results.py \
  ga_mdp_leadingones_mutation_only.csv \
  ga_mdp_leadingones.csv \
  --labels \
      'Mutation only ($p_c=0$)' \
      'Crossover ($p_c=0.9$)' \
  --output-prefix leadingones_combined \
  --title "LeadingOnes: exact population-MDP predictions versus simulation"

# Performance Prediction Comparison for Jump Problem

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem jump \
  --crossover-rate 0.9 \
  --output-prefix ga_mdp_jump

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem jump \
  --crossover-rate 0.0 \
  --output-prefix ga_mdp_jump_mutation_only

python plot_combined_ga_results.py \
  ga_mdp_jump_mutation_only.csv \
  ga_mdp_jump.csv \
  --labels \
      'Mutation only ($p_c=0$)' \
      'Crossover ($p_c=0.9$)' \
  --output-prefix jump_combined \
  --title "Jump: exact population-MDP predictions versus simulation"

# Performance Prediction Comparison for Trap Problem

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem trap \
  --crossover-rate 0.9 \
  --output-prefix ga_mdp_trap

python population_mdp_test_1.py \
  --n 4 \
  --mu 3 \
  --generations 30 \
  --runs 50000 \
  --problem trap \
  --crossover-rate 0.0 \
  --output-prefix ga_mdp_trap_mutation_only

python plot_combined_ga_results.py \
  ga_mdp_trap_mutation_only.csv \
  ga_mdp_trap.csv \
  --labels \
      'Mutation only ($p_c=0$)' \
      'Crossover ($p_c=0.9$)' \
  --output-prefix trap_combined \
  --title "Trap: exact population-MDP predictions versus simulation"