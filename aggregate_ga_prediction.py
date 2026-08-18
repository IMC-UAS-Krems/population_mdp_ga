#!/usr/bin/env python3
"""Stage II experiment: aggregate-state prediction for genetic algorithms.

The script evaluates whether aggregate fitness/diversity states predict GA
behaviour when exact multiset population states are intractable.

Protocol
--------
1. Generate independent training, validation, and test trajectories.
2. Fit descriptor quantizers on training data only.
3. Estimate an aggregate Markov transition matrix from training transitions.
4. Use validation trajectories to select the number of quantile bins.
5. Predict test-set trajectories from the aggregate initial-state distribution.
6. Compare predicted and independently simulated trajectories for:
   - expected best fitness,
   - expected normalized pairwise diversity,
   - finite-horizon success probability.
7. Compare nested aggregate descriptors:
   fitness, fitness_var, fitness_diversity, and crossover_sensitive.

Default GA
----------
Steady-state (mu+1) GA with tournament selection, uniform crossover,
independent bit mutation, and elitist deletion of one minimum-fitness member.

Example
-------
python aggregate_ga_prediction.py \
    --problem onemax --n 20 --mu 16 --generations 100 \
    --train-runs 10000 --validation-runs 2000 --test-runs 10000 \
    --output-prefix aggregate_onemax
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DESCRIPTOR_SETS = {
    "fitness": ("best", "mean"),
    "fitness_var": ("best", "mean", "variance"),
    "fitness_diversity": ("best", "mean", "variance", "diversity"),
    "crossover_sensitive": (
        "best", "mean", "variance", "diversity", "allele_entropy",
        "selected_parent_distance",
    ),
}


@dataclass(frozen=True)
class Config:
    problem: str = "onemax"
    n: int = 20
    mu: int = 16
    generations: int = 100
    train_runs: int = 10_000
    validation_runs: int = 2_000
    test_runs: int = 10_000
    tournament_k: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    jump_k: int = 4
    trap_k: int = 4
    nk_k: int = 3
    seed: int = 20260818
    candidate_bins: Tuple[int, ...] = (4, 6, 8, 10)
    smoothing: float = 0.0
    output_prefix: str = "aggregate_ga_prediction"

    @property
    def pm(self) -> float:
        return 1.0 / self.n if self.mutation_rate is None else self.mutation_rate


@dataclass
class TrajectoryData:
    descriptors: np.ndarray  # runs x (T+1) x d_all
    target_hit: np.ndarray   # runs x (T+1), cumulative success indicator


@dataclass
class Quantizer:
    feature_names: Tuple[str, ...]
    edges: Tuple[np.ndarray, ...]

    def transform_one(self, vector: np.ndarray) -> Tuple[int, ...]:
        return tuple(
            int(np.searchsorted(edge, value, side="right"))
            for edge, value in zip(self.edges, vector)
        )

    def transform(self, matrix: np.ndarray) -> List[Tuple[int, ...]]:
        return [self.transform_one(row) for row in matrix]


@dataclass
class AggregateModel:
    state_to_index: Dict[Tuple[int, ...], int]
    states: List[Tuple[int, ...]]
    transition: csr_matrix
    best_readout: np.ndarray
    diversity_readout: np.ndarray
    target_readout: np.ndarray
    unknown_state: Tuple[int, ...]


def bitcount(x: int) -> int:
    return int(x.bit_count())


def make_problem(cfg: Config):
    n = cfg.n
    if cfg.problem == "onemax":
        fitness = lambda x: float(bitcount(x))
        optimum = float(n)
    elif cfg.problem == "leadingones":
        def fitness(x: int) -> float:
            bits = format(x, f"0{n}b")
            return float(len(bits) - len(bits.lstrip("1")))
        optimum = float(n)
    elif cfg.problem == "jump":
        k = cfg.jump_k
        if not 1 <= k < n:
            raise ValueError("jump_k must satisfy 1 <= jump_k < n")
        def fitness(x: int) -> float:
            u = bitcount(x)
            if u == n or u <= n - k:
                return float(u)
            return float(n - u)
        optimum = float(n)
    elif cfg.problem == "trap":
        k = cfg.trap_k
        if n % k:
            raise ValueError("n must be divisible by trap_k")
        mask = (1 << k) - 1
        def fitness(x: int) -> float:
            total = 0.0
            for block in range(n // k):
                u = bitcount((x >> (block * k)) & mask)
                total += k if u == k else k - 1 - u
            return total
        optimum = float(n)
    elif cfg.problem == "nk":
        rng = np.random.default_rng(cfg.seed + 991)
        k = cfg.nk_k
        if not 0 <= k < n:
            raise ValueError("nk_k must satisfy 0 <= nk_k < n")
        tables = rng.random((n, 1 << (k + 1)))
        neighborhoods = [tuple((i + j) % n for j in range(k + 1)) for i in range(n)]
        def fitness(x: int) -> float:
            total = 0.0
            for i, positions in enumerate(neighborhoods):
                pattern = 0
                for j, pos in enumerate(positions):
                    pattern |= ((x >> pos) & 1) << j
                total += tables[i, pattern]
            return float(total)
        # For n <= 22, exact optimum is feasible; otherwise use maximum table sum as target proxy.
        if n <= 22:
            optimum = max(fitness(x) for x in range(1 << n))
        else:
            optimum = float("inf")
    else:
        raise ValueError(f"Unknown problem {cfg.problem}")
    return fitness, optimum


def tournament(pop: Sequence[int], values: np.ndarray, k: int, rng: np.random.Generator) -> int:
    positions = rng.integers(0, len(pop), size=k)
    sampled_values = values[positions]
    winners = positions[sampled_values == sampled_values.max()]
    return int(pop[int(rng.choice(winners))])


def uniform_crossover(p1: int, p2: int, n: int, pc: float, rng: np.random.Generator) -> int:
    if rng.random() >= pc:
        return p1 if rng.random() < 0.5 else p2
    mask = int(rng.integers(0, 1 << n))
    full = (1 << n) - 1
    return (p1 & mask) | (p2 & (~mask & full))


def mutate(x: int, n: int, pm: float, rng: np.random.Generator) -> int:
    for bit in range(n):
        if rng.random() < pm:
            x ^= 1 << bit
    return x


def normalized_pairwise_diversity(pop: Sequence[int], n: int) -> float:
    mu = len(pop)
    if mu < 2:
        return 0.0
    counts = Counter(pop)
    total = 0.0
    genotypes = list(counts)
    for i, x in enumerate(genotypes):
        for y in genotypes[i + 1:]:
            total += counts[x] * counts[y] * bitcount(x ^ y)
    return total / (n * math.comb(mu, 2))


def allele_entropy(pop: Sequence[int], n: int) -> float:
    frequencies = np.zeros(n, dtype=float)
    for x in pop:
        for bit in range(n):
            frequencies[bit] += (x >> bit) & 1
    frequencies /= len(pop)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(
            (frequencies == 0.0) | (frequencies == 1.0),
            0.0,
            -frequencies * np.log2(frequencies)
            - (1.0 - frequencies) * np.log2(1.0 - frequencies),
        )
    return float(entropy.mean())


def tournament_winner_probabilities(values: np.ndarray, k: int) -> np.ndarray:
    """Probability that each population position wins a tournament.

    Sampling is with replacement and ties are uniform among sampled winning
    positions. This exact calculation is O(mu^k), appropriate for small k.
    """
    mu = len(values)
    probabilities = np.zeros(mu, dtype=float)
    for positions in np.ndindex(*(mu,) * k):
        p = (1.0 / mu) ** k
        sampled = values[list(positions)]
        best = sampled.max()
        winning_slots = [slot for slot, pos in enumerate(positions) if values[pos] == best]
        for slot in winning_slots:
            probabilities[positions[slot]] += p / len(winning_slots)
    return probabilities


def expected_selected_parent_distance(pop: Sequence[int], values: np.ndarray, n: int, k: int) -> float:
    probs = tournament_winner_probabilities(values, k)
    total = 0.0
    for i, x in enumerate(pop):
        for j, y in enumerate(pop):
            total += probs[i] * probs[j] * bitcount(x ^ y) / n
    return float(total)


def population_descriptors(pop: Sequence[int], fitness, cfg: Config) -> np.ndarray:
    values = np.asarray([fitness(x) for x in pop], dtype=float)
    return np.asarray([
        values.max(),
        values.mean(),
        values.var(),
        normalized_pairwise_diversity(pop, cfg.n),
        allele_entropy(pop, cfg.n),
        expected_selected_parent_distance(pop, values, cfg.n, cfg.tournament_k),
    ], dtype=float)


def simulate_dataset(cfg: Config, runs: int, seed: int) -> TrajectoryData:
    fitness, optimum = make_problem(cfg)
    rng = np.random.default_rng(seed)
    d = len(DESCRIPTOR_SETS["crossover_sensitive"])
    descriptors = np.empty((runs, cfg.generations + 1, d), dtype=np.float32)
    target_hit = np.zeros((runs, cfg.generations + 1), dtype=np.float32)

    for run in range(runs):
        pop = [int(rng.integers(0, 1 << cfg.n)) for _ in range(cfg.mu)]
        ever_hit = False
        for t in range(cfg.generations + 1):
            desc = population_descriptors(pop, fitness, cfg)
            descriptors[run, t] = desc
            if math.isfinite(optimum) and desc[0] >= optimum - 1e-12:
                ever_hit = True
            target_hit[run, t] = float(ever_hit)
            if t == cfg.generations:
                break

            values = np.asarray([fitness(x) for x in pop], dtype=float)
            p1 = tournament(pop, values, cfg.tournament_k, rng)
            p2 = tournament(pop, values, cfg.tournament_k, rng)
            child = uniform_crossover(p1, p2, cfg.n, cfg.crossover_rate, rng)
            child = mutate(child, cfg.n, cfg.pm, rng)

            enlarged = pop + [child]
            enlarged_values = np.asarray([fitness(x) for x in enlarged], dtype=float)
            worst = np.flatnonzero(enlarged_values == enlarged_values.min())
            delete_position = int(rng.choice(worst))
            del enlarged[delete_position]
            pop = enlarged

    return TrajectoryData(descriptors=descriptors, target_hit=target_hit)


def feature_indices(feature_names: Sequence[str]) -> List[int]:
    all_names = DESCRIPTOR_SETS["crossover_sensitive"]
    return [all_names.index(name) for name in feature_names]


def fit_quantizer(training: np.ndarray, feature_names: Tuple[str, ...], bins: int) -> Quantizer:
    indices = feature_indices(feature_names)
    flat = training[:, :, indices].reshape(-1, len(indices))
    edges = []
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for j in range(flat.shape[1]):
        candidate = np.quantile(flat[:, j], quantiles)
        candidate = np.unique(candidate[np.isfinite(candidate)])
        edges.append(candidate)
    return Quantizer(feature_names=feature_names, edges=tuple(edges))


def encode_trajectories(data: np.ndarray, quantizer: Quantizer) -> List[List[Tuple[int, ...]]]:
    indices = feature_indices(quantizer.feature_names)
    encoded = []
    for run in range(data.shape[0]):
        encoded.append(quantizer.transform(data[run, :, indices]))
    return encoded


def fit_aggregate_model(
    training: TrajectoryData,
    quantizer: Quantizer,
    smoothing: float = 0.0,
) -> AggregateModel:
    encoded = encode_trajectories(training.descriptors, quantizer)
    observed_states = sorted({state for trajectory in encoded for state in trajectory})
    unknown = tuple([-1] * len(quantizer.feature_names))
    states = observed_states + [unknown]
    state_to_index = {state: i for i, state in enumerate(states)}
    unknown_index = state_to_index[unknown]

    transition_counts: Dict[int, Counter] = defaultdict(Counter)
    best_sum = np.zeros(len(states), dtype=float)
    diversity_sum = np.zeros(len(states), dtype=float)
    target_sum = np.zeros(len(states), dtype=float)
    visits = np.zeros(len(states), dtype=float)

    for run, trajectory in enumerate(encoded):
        for t, state in enumerate(trajectory):
            i = state_to_index[state]
            best_sum[i] += training.descriptors[run, t, 0]
            diversity_sum[i] += training.descriptors[run, t, 3]
            target_sum[i] += training.target_hit[run, t]
            visits[i] += 1
            if t + 1 < len(trajectory):
                j = state_to_index[trajectory[t + 1]]
                transition_counts[i][j] += 1

    global_best = float(training.descriptors[:, :, 0].mean())
    global_diversity = float(training.descriptors[:, :, 3].mean())
    global_target = float(training.target_hit.mean())
    best_readout = np.divide(best_sum, visits, out=np.full_like(best_sum, global_best), where=visits > 0)
    diversity_readout = np.divide(
        diversity_sum, visits, out=np.full_like(diversity_sum, global_diversity), where=visits > 0
    )
    target_readout = np.divide(target_sum, visits, out=np.full_like(target_sum, global_target), where=visits > 0)

    rows, cols, values = [], [], []
    for i in range(len(states)):
        counts = transition_counts.get(i, Counter())
        total = float(sum(counts.values()))
        if i == unknown_index or total == 0.0:
            rows.append(i); cols.append(i); values.append(1.0)
            continue
        if smoothing > 0.0:
            # Sparse empirical-Bayes smoothing toward a self-loop.
            counts = counts.copy()
            counts[i] += smoothing
            total += smoothing
        for j, count in counts.items():
            rows.append(i); cols.append(j); values.append(count / total)

    transition = csr_matrix((values, (rows, cols)), shape=(len(states), len(states)))
    return AggregateModel(
        state_to_index=state_to_index,
        states=states,
        transition=transition,
        best_readout=best_readout,
        diversity_readout=diversity_readout,
        target_readout=target_readout,
        unknown_state=unknown,
    )


def aggregate_initial_distribution(
    data: TrajectoryData, quantizer: Quantizer, model: AggregateModel
) -> Tuple[np.ndarray, float]:
    indices = feature_indices(quantizer.feature_names)
    states = quantizer.transform(data.descriptors[:, 0, indices])
    distribution = np.zeros(len(model.states), dtype=float)
    unknown_count = 0
    observed_states = [state for state in model.states if state != model.unknown_state]
    observed_array = np.asarray(observed_states, dtype=float)
    for state in states:
        i = model.state_to_index.get(state)
        if i is None:
            # Robust out-of-sample backoff: map an unseen quantized descriptor
            # to its nearest observed aggregate state in normalized bin space.
            # This avoids an artificial absorbing unknown state.
            unknown_count += 1
            query = np.asarray(state, dtype=float)
            scale = np.maximum(1.0, observed_array.max(axis=0) - observed_array.min(axis=0))
            distances = np.sum(np.abs(observed_array - query) / scale, axis=1)
            nearest = observed_states[int(np.argmin(distances))]
            i = model.state_to_index[nearest]
        distribution[i] += 1
    distribution /= len(states)
    return distribution, unknown_count / len(states)


def predict_trajectory(
    initial: np.ndarray, model: AggregateModel, generations: int
) -> Dict[str, np.ndarray]:
    distribution = initial.copy()
    result = {
        "best": np.empty(generations + 1),
        "diversity": np.empty(generations + 1),
        "success": np.empty(generations + 1),
    }
    for t in range(generations + 1):
        result["best"][t] = distribution @ model.best_readout
        result["diversity"][t] = distribution @ model.diversity_readout
        result["success"][t] = distribution @ model.target_readout
        distribution = distribution @ model.transition
    # Success is a cumulative probability; monotonic repair counters finite-sample readout noise.
    result["success"] = np.maximum.accumulate(np.clip(result["success"], 0.0, 1.0))
    return result


def empirical_trajectory(data: TrajectoryData) -> Dict[str, np.ndarray]:
    return {
        "best": data.descriptors[:, :, 0].mean(axis=0),
        "diversity": data.descriptors[:, :, 3].mean(axis=0),
        "success": data.target_hit.mean(axis=0),
    }


def trajectory_metrics(prediction: np.ndarray, observed: np.ndarray) -> Dict[str, float]:
    error = prediction - observed
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_error": float(error[-1]),
    }


def evaluate_candidate(
    training: TrajectoryData,
    evaluation: TrajectoryData,
    feature_names: Tuple[str, ...],
    bins: int,
    smoothing: float,
) -> Tuple[float, Quantizer, AggregateModel, Dict[str, np.ndarray], float]:
    quantizer = fit_quantizer(training.descriptors, feature_names, bins)
    model = fit_aggregate_model(training, quantizer, smoothing)
    initial, unknown_rate = aggregate_initial_distribution(evaluation, quantizer, model)
    prediction = predict_trajectory(initial, model, evaluation.descriptors.shape[1] - 1)
    observed = empirical_trajectory(evaluation)
    # Scale-normalized validation score across best fitness and diversity.
    span_best = max(1.0, float(np.ptp(observed["best"])))
    score = (
        trajectory_metrics(prediction["best"], observed["best"])["rmse"] / span_best
        + trajectory_metrics(prediction["diversity"], observed["diversity"])["rmse"]
    )
    return score, quantizer, model, prediction, unknown_rate


def save_results(
    cfg: Config,
    test: TrajectoryData,
    results: Mapping[str, Mapping],
) -> None:
    prefix = Path(cfg.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    observed = empirical_trajectory(test)
    generations = np.arange(cfg.generations + 1)

    trajectory_rows = []
    metric_rows = []
    for descriptor_name, item in results.items():
        prediction = item["prediction"]
        for t in generations:
            trajectory_rows.append({
                "descriptor": descriptor_name,
                "generation": int(t),
                "predicted_best": prediction["best"][t],
                "observed_best": observed["best"][t],
                "predicted_diversity": prediction["diversity"][t],
                "observed_diversity": observed["diversity"][t],
                "predicted_success": prediction["success"][t],
                "observed_success": observed["success"][t],
            })
        for target in ("best", "diversity", "success"):
            metrics = trajectory_metrics(prediction[target], observed[target])
            metric_rows.append({
                "descriptor": descriptor_name,
                "target": target,
                "selected_bins": item["bins"],
                "aggregate_states": len(item["model"].states),
                "initial_unknown_rate": item["unknown_rate"],
                **metrics,
            })

    pd.DataFrame(trajectory_rows).to_csv(
        prefix.with_name(prefix.name + "_trajectories").with_suffix(".csv"), index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        prefix.with_name(prefix.name + "_metrics").with_suffix(".csv"), index=False
    )
    metadata = {
        "config": {**asdict(cfg), "effective_mutation_rate": cfg.pm},
        "descriptor_selection": {
            name: {
                "features": list(DESCRIPTOR_SETS[name]),
                "selected_bins": item["bins"],
                "validation_score": item["validation_score"],
                "aggregate_states": len(item["model"].states),
                "initial_unknown_rate": item["unknown_rate"],
            }
            for name, item in results.items()
        },
    }
    prefix.with_name(prefix.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    targets = (
        ("best", "Expected best fitness"),
        ("diversity", "Expected normalized diversity"),
        ("success", "Success probability"),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (target, ylabel) in zip(axes, targets):
        ax.plot(generations, observed[target], color="black", linewidth=2.5, label="Independent test simulation")
        for i, (descriptor_name, item) in enumerate(results.items()):
            ax.plot(
                generations,
                item["prediction"][target],
                linewidth=1.8,
                color=colors[i % len(colors)],
                label=descriptor_name.replace("_", " "),
            )
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{cfg.problem}: aggregate-state predictions, n={cfg.n}, mu={cfg.mu}, pc={cfg.crossover_rate}"
    )
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(cfg: Config) -> None:
    print("Generating independent trajectory splits...")
    training = simulate_dataset(cfg, cfg.train_runs, cfg.seed + 1)
    validation = simulate_dataset(cfg, cfg.validation_runs, cfg.seed + 2)
    test = simulate_dataset(cfg, cfg.test_runs, cfg.seed + 3)

    results = {}
    for descriptor_name, features in DESCRIPTOR_SETS.items():
        candidate_records = []
        for bins in cfg.candidate_bins:
            score, quantizer, model, prediction, unknown_rate = evaluate_candidate(
                training, validation, features, bins, cfg.smoothing
            )
            candidate_records.append((score, bins, quantizer, model, unknown_rate))
            print(
                f"{descriptor_name:22s} bins={bins:2d} "
                f"validation_score={score:.6f} states={len(model.states)}"
            )
        score, bins, quantizer, model, _ = min(candidate_records, key=lambda record: record[0])
        initial, unknown_rate = aggregate_initial_distribution(test, quantizer, model)
        prediction = predict_trajectory(initial, model, cfg.generations)
        results[descriptor_name] = {
            "validation_score": score,
            "bins": bins,
            "quantizer": quantizer,
            "model": model,
            "prediction": prediction,
            "unknown_rate": unknown_rate,
        }

    save_results(cfg, test, results)
    metrics_path = Path(cfg.output_prefix).with_name(
        Path(cfg.output_prefix).name + "_metrics"
    ).with_suffix(".csv")
    print(pd.read_csv(metrics_path).to_string(index=False))
    print(f"Outputs written with prefix {cfg.output_prefix}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["onemax", "leadingones", "jump", "trap", "nk"], default="onemax")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--mu", type=int, default=16)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--train-runs", type=int, default=10000)
    parser.add_argument("--validation-runs", type=int, default=2000)
    parser.add_argument("--test-runs", type=int, default=10000)
    parser.add_argument("--tournament-k", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--jump-k", type=int, default=4)
    parser.add_argument("--trap-k", type=int, default=4)
    parser.add_argument("--nk-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--candidate-bins", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="aggregate_ga_prediction")
    a = parser.parse_args()
    return Config(
        problem=a.problem, n=a.n, mu=a.mu, generations=a.generations,
        train_runs=a.train_runs, validation_runs=a.validation_runs,
        test_runs=a.test_runs, tournament_k=a.tournament_k,
        crossover_rate=a.crossover_rate, mutation_rate=a.mutation_rate,
        jump_k=a.jump_k, trap_k=a.trap_k, nk_k=a.nk_k, seed=a.seed,
        candidate_bins=tuple(a.candidate_bins), smoothing=a.smoothing,
        output_prefix=a.output_prefix,
    )


if __name__ == "__main__":
    main(parse_args())
