python diversity_crossover_interaction.py \
    --problem jump \
    --n 20 \
    --mu 16 \
    --jump-k 4 \
    --matched-pairs 100 \
    --rollouts 1000 \
    --horizons 25 50 100 \
    --minimum-diversity-gap 0.10 \
    --output-prefix jump_diversity_crossover