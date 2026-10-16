# Population MDPs for Genetic Algorithms

Source code for all experiments in the paper *Population Markov Decision Processes: A Theoretical Framework for Performance Analysis and Prediction for Genetic Algorithms*.

## Requirements

Python 3.12 with `numpy`, `scipy`, `pandas` and `matplotlib`:

```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy pandas matplotlib
```

## Experiments

| Paper section | Script(s) | Runner | Results |
|---|---|---|---|
| 7.1 Exact population-MDP vs. Monte Carlo (OneMax, LeadingOnes, Jump, Trap; Table 2, Fig. 1) | `population_mdp_test_1.py`, `plot_combined_ga_results.py` | `test_1.sh` | `results/test_1/` |
| 7.2 Aggregate-state prediction (OneMax; Table 4, Fig. 2) | `compare_aggregate_models.py`, `rich_aggregate_ga_prediction.py`, `aggregate_ga_prediction.py` | `aggregate_predictions.sh` | `results/aggregate_predictions/` |
| 7.3 Diversity–crossover interaction (Jump; Fig. 3) | `diversity_crossover_interaction.py` | `diversity_crossover_test.sh` | `results/diversity_crossover/` |

Each runner script contains the exact parameters used in the paper:

```bash
bash test_1.sh
bash aggregate_predictions.sh
bash diversity_crossover_test.sh
```

Every Python script accepts `--help` for the full list of options.

## License

MIT, see [LICENSE](LICENSE).
