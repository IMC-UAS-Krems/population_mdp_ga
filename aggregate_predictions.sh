# python rich_aggregate_ga_prediction.py \
#     --problem onemax \
#     --n 20 \
#     --mu 16 \
#     --generations 100 \
#     --train-runs 10000 \
#     --validation-runs 2000 \
#     --test-runs 10000 \
#     --tournament-k 2 \
#     --crossover-rate 0.9 \
#     --candidate-bins 3 4 5 \
#     --candidate-windows 0 10 20 \
#     --allele-hist-bins 5 \
#     --fitness-depth 3 \
#     --output-prefix rich_onemax

python compare_aggregate_models.py \
    --problem onemax \
    --n 20 \
    --mu 16 \
    --generations 100 \
    --train-runs 10000 \
    --validation-runs 2000 \
    --test-runs 10000 \
    --tournament-k 2 \
    --crossover-rate 0.9 \
    --candidate-bins 3 4 5 \
    --candidate-windows 0 10 20 \
    --allele-hist-bins 5 \
    --fitness-depth 3 \
    --output-prefix onemax_model_comparison