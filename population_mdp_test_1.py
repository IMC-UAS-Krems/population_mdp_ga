#!/usr/bin/env python3
"""Exact population-MDP vs Monte Carlo GA trajectories.

First experiment for the paper:
  - binary pseudo-Boolean objective (default: OneMax)
  - fixed-size (mu+1) steady-state GA
  - tournament selection with replacement
  - uniform crossover (or mutation-only ablation)
  - independent bit mutation
  - elitist replacement with uniform tie-breaking

The script enumerates the exact multiset population state space, constructs
its sparse transition matrix, computes E[best fitness at generation t], and
compares the theoretical trajectory with independently simulated runs.

Example:
    python ga_population_mdp_experiment.py --n 4 --mu 3 --runs 50000
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

Genotype = int
Population = Tuple[int, ...]  # multiplicities indexed by genotype


@dataclass(frozen=True)
class Config:
    n: int = 4
    mu: int = 3
    generations: int = 30
    runs: int = 50_000
    tournament_k: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    problem: str = "onemax"
    jump_k: int = 2
    trap_k: int = 2
    seed: int = 20260817
    output_prefix: str = "ga_mdp_onemax"

    @property
    def pm(self) -> float:
        return 1.0 / self.n if self.mutation_rate is None else self.mutation_rate


def bitcount(x: int) -> int:
    return int(x.bit_count())


def make_fitness(cfg: Config) -> Callable[[int], float]:
    n = cfg.n
    if cfg.problem == "onemax":
        return lambda x: float(bitcount(x))
    if cfg.problem == "leadingones":
        def leading_ones(x: int) -> float:
            bits = format(x, f"0{n}b")
            return float(len(bits) - len(bits.lstrip("1")))
        return leading_ones
    if cfg.problem == "jump":
        k = cfg.jump_k
        if not (1 <= k < n):
            raise ValueError("jump_k must satisfy 1 <= jump_k < n")
        def jump(x: int) -> float:
            u = bitcount(x)
            if u == n or u <= n - k:
                return float(u)
            return float(n - u)
        return jump
    if cfg.problem == "trap":
        k = cfg.trap_k
        if n % k != 0:
            raise ValueError("For concatenated trap, n must be divisible by trap_k")
        mask = (1 << k) - 1
        def trap(x: int) -> float:
            total = 0.0
            for block in range(n // k):
                u = bitcount((x >> (block * k)) & mask)
                total += k if u == k else (k - 1 - u)
            return total
        return trap
    raise ValueError(f"Unknown problem: {cfg.problem}")


def weak_compositions(total: int, parts: int) -> Iterable[Population]:
    """Yield all weak compositions of total into parts entries."""
    for bars in itertools.combinations(range(total + parts - 1), parts - 1):
        prev = -1
        out = []
        for b in bars + (total + parts - 1,):
            out.append(b - prev - 1)
            prev = b
        yield tuple(out)


def enumerate_populations(n: int, mu: int) -> List[Population]:
    return list(weak_compositions(mu, 1 << n))


def population_best(pop: Population, fitness_values: np.ndarray) -> float:
    present = np.flatnonzero(np.asarray(pop))
    return float(np.max(fitness_values[present]))


def population_diversity(pop: Population, n: int) -> float:
    """Normalized average pairwise Hamming distance; 0 for mu < 2."""
    mu = sum(pop)
    if mu < 2:
        return 0.0
    total = 0.0
    for x, cx in enumerate(pop):
        if cx == 0:
            continue
        for y in range(x + 1, len(pop)):
            cy = pop[y]
            if cy:
                total += cx * cy * bitcount(x ^ y)
    return total / (n * math.comb(mu, 2))


def tournament_winner_distribution(
    pop: Population, fitness_values: np.ndarray, k: int
) -> Dict[Genotype, float]:
    """Exact winner distribution for k-way tournament sampling positions with replacement.

    Ties among sampled positions with maximum fitness are broken uniformly.
    Enumeration is inexpensive for the small exact-state experiments.
    """
    mu = sum(pop)
    genotypes = [x for x, c in enumerate(pop) if c]
    draw_prob = {x: pop[x] / mu for x in genotypes}
    winner = defaultdict(float)
    for sample in itertools.product(genotypes, repeat=k):
        p_sample = math.prod(draw_prob[x] for x in sample)
        max_fit = max(fitness_values[x] for x in sample)
        winning_positions = [x for x in sample if fitness_values[x] == max_fit]
        for x in winning_positions:
            winner[x] += p_sample / len(winning_positions)
    return dict(winner)


def uniform_crossover_distribution(
    p1: int, p2: int, n: int, pc: float
) -> Dict[Genotype, float]:
    """One-child uniform crossover; no-crossover copies one random parent."""
    out = defaultdict(float)
    out[p1] += (1.0 - pc) / 2.0
    out[p2] += (1.0 - pc) / 2.0
    for mask in range(1 << n):
        child = (p1 & mask) | (p2 & (~mask & ((1 << n) - 1)))
        out[child] += pc / (1 << n)
    return dict(out)


def mutation_distribution(x: int, n: int, pm: float) -> Dict[Genotype, float]:
    out: Dict[Genotype, float] = {}
    for y in range(1 << n):
        d = bitcount(x ^ y)
        out[y] = (pm ** d) * ((1.0 - pm) ** (n - d))
    return out


def offspring_distribution(
    pop: Population,
    fitness_values: np.ndarray,
    cfg: Config,
    mutation_cache: Sequence[Mapping[int, float]],
) -> Dict[Genotype, float]:
    select = tournament_winner_distribution(pop, fitness_values, cfg.tournament_k)
    pre_mutation = defaultdict(float)
    for p1, q1 in select.items():
        for p2, q2 in select.items():
            for child, qc in uniform_crossover_distribution(
                p1, p2, cfg.n, cfg.crossover_rate
            ).items():
                pre_mutation[child] += q1 * q2 * qc
    offspring = defaultdict(float)
    for child, qc in pre_mutation.items():
        for mutant, qm in mutation_cache[child].items():
            offspring[mutant] += qc * qm
    return dict(offspring)


def replacement_distribution(
    pop: Population, child: int, fitness_values: np.ndarray
) -> Dict[Population, float]:
    """Elitist (mu+1) replacement; delete one minimum-fitness position uniformly."""
    enlarged = list(pop)
    enlarged[child] += 1
    present = [x for x, c in enumerate(enlarged) if c]
    worst_fitness = min(fitness_values[x] for x in present)
    worst = [x for x in present if fitness_values[x] == worst_fitness]
    denominator = sum(enlarged[x] for x in worst)
    out = defaultdict(float)
    for x in worst:
        next_pop = enlarged.copy()
        next_pop[x] -= 1
        out[tuple(next_pop)] += enlarged[x] / denominator
    return dict(out)


def build_transition_matrix(
    states: Sequence[Population], fitness_values: np.ndarray, cfg: Config
) -> csr_matrix:
    index = {p: i for i, p in enumerate(states)}
    mutation_cache = [mutation_distribution(x, cfg.n, cfg.pm) for x in range(1 << cfg.n)]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, pop in enumerate(states):
        row = defaultdict(float)
        for child, p_child in offspring_distribution(
            pop, fitness_values, cfg, mutation_cache
        ).items():
            for next_pop, p_rep in replacement_distribution(
                pop, child, fitness_values
            ).items():
                row[index[next_pop]] += p_child * p_rep
        row_sum = sum(row.values())
        if not np.isclose(row_sum, 1.0, atol=1e-12):
            raise RuntimeError(f"Transition row {i} sums to {row_sum}")
        for j, probability in row.items():
            if probability > 0.0:
                rows.append(i)
                cols.append(j)
                data.append(probability)
    K = csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    return K


def iid_uniform_initial_distribution(states: Sequence[Population], n: int, mu: int) -> np.ndarray:
    """Distribution of an unordered population generated by iid uniform genotypes."""
    m = 1 << n
    denominator = float(m ** mu)
    dist = np.empty(len(states), dtype=float)
    mu_factorial = math.factorial(mu)
    for i, pop in enumerate(states):
        multiplicity = mu_factorial
        for c in pop:
            multiplicity //= math.factorial(c)
        dist[i] = multiplicity / denominator
    if not np.isclose(dist.sum(), 1.0, atol=1e-12):
        raise RuntimeError(f"Initial distribution sums to {dist.sum()}")
    return dist


def theoretical_trajectory(
    K: csr_matrix,
    initial: np.ndarray,
    score: np.ndarray,
    generations: int,
) -> np.ndarray:
    distribution = initial.copy()
    values = np.empty(generations + 1)
    for t in range(generations + 1):
        values[t] = float(distribution @ score)
        distribution = distribution @ K
    return values


def sample_tournament(pop: List[int], fitness: Callable[[int], float], k: int, rng: np.random.Generator) -> int:
    sample = [pop[int(rng.integers(len(pop)))] for _ in range(k)]
    values = np.asarray([fitness(x) for x in sample])
    best_positions = np.flatnonzero(values == values.max())
    return sample[int(rng.choice(best_positions))]


def sample_child(pop: List[int], fitness: Callable[[int], float], cfg: Config, rng: np.random.Generator) -> int:
    p1 = sample_tournament(pop, fitness, cfg.tournament_k, rng)
    p2 = sample_tournament(pop, fitness, cfg.tournament_k, rng)
    if rng.random() < cfg.crossover_rate:
        mask = int(rng.integers(1 << cfg.n))
        child = (p1 & mask) | (p2 & (~mask & ((1 << cfg.n) - 1)))
    else:
        child = p1 if rng.random() < 0.5 else p2
    for bit in range(cfg.n):
        if rng.random() < cfg.pm:
            child ^= 1 << bit
    return child


def simulate(cfg: Config, fitness: Callable[[int], float]) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    trajectories = np.empty((cfg.runs, cfg.generations + 1), dtype=float)
    for run in range(cfg.runs):
        pop = [int(rng.integers(1 << cfg.n)) for _ in range(cfg.mu)]
        trajectories[run, 0] = max(fitness(x) for x in pop)
        for t in range(1, cfg.generations + 1):
            child = sample_child(pop, fitness, cfg, rng)
            enlarged = pop + [child]
            values = np.asarray([fitness(x) for x in enlarged])
            worst_positions = np.flatnonzero(values == values.min())
            del enlarged[int(rng.choice(worst_positions))]
            pop = enlarged
            trajectories[run, t] = max(fitness(x) for x in pop)
    mean = trajectories.mean(axis=0)
    se = trajectories.std(axis=0, ddof=1) / math.sqrt(cfg.runs)
    return mean, se


def run(cfg: Config) -> None:
    fitness = make_fitness(cfg)
    fitness_values = np.asarray([fitness(x) for x in range(1 << cfg.n)], dtype=float)
    states = enumerate_populations(cfg.n, cfg.mu)
    expected_states = math.comb((1 << cfg.n) + cfg.mu - 1, cfg.mu)
    if len(states) != expected_states:
        raise RuntimeError("Population enumeration failed")

    print(f"Exact states: {len(states)}")
    K = build_transition_matrix(states, fitness_values, cfg)
    row_sums = np.asarray(K.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise RuntimeError("Transition matrix is not row stochastic")

    initial = iid_uniform_initial_distribution(states, cfg.n, cfg.mu)
    best_scores = np.asarray([population_best(p, fitness_values) for p in states])
    diversity_scores = np.asarray([population_diversity(p, cfg.n) for p in states])

    theory_best = theoretical_trajectory(K, initial, best_scores, cfg.generations)
    theory_diversity = theoretical_trajectory(K, initial, diversity_scores, cfg.generations)
    empirical_best, empirical_se = simulate(cfg, fitness)

    generations = np.arange(cfg.generations + 1)
    error = empirical_best - theory_best
    rmse = float(np.sqrt(np.mean(error ** 2)))
    max_abs_error = float(np.max(np.abs(error)))

    prefix = Path(cfg.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "generation", "theory_expected_best", "mc_mean_best",
            "mc_se", "mc_ci95_lower", "mc_ci95_upper", "theory_expected_diversity"
        ])
        for t in generations:
            writer.writerow([
                int(t), theory_best[t], empirical_best[t], empirical_se[t],
                empirical_best[t] - 1.96 * empirical_se[t],
                empirical_best[t] + 1.96 * empirical_se[t],
                theory_diversity[t],
            ])

    summary = {
        "config": {**asdict(cfg), "effective_mutation_rate": cfg.pm},
        "number_of_exact_states": len(states),
        "transition_nonzeros": int(K.nnz),
        "rmse_mc_vs_theory": rmse,
        "max_absolute_error_mc_vs_theory": max_abs_error,
    }
    json_path = prefix.with_name(prefix.name + "_summary").with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(generations, theory_best, label="Population-MDP prediction", linewidth=2)
    ax.plot(generations, empirical_best, label="Monte Carlo mean", linestyle="--", linewidth=1.8)
    ax.fill_between(
        generations,
        empirical_best - 1.96 * empirical_se,
        empirical_best + 1.96 * empirical_se,
        alpha=0.25,
        label="Monte Carlo 95% CI",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Expected best fitness")
    ax.set_title(f"{cfg.problem}: exact population-MDP vs simulation")
    ax.legend()
    fig.tight_layout()
    png_path = prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}, {json_path}, and {png_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--runs", type=int, default=50_000)
    parser.add_argument("--tournament-k", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--problem", choices=["onemax", "leadingones", "jump", "trap"], default="onemax")
    parser.add_argument("--jump-k", type=int, default=2)
    parser.add_argument("--trap-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-prefix", default="ga_mdp_onemax")
    a = parser.parse_args()
    return Config(
        n=a.n, mu=a.mu, generations=a.generations, runs=a.runs,
        tournament_k=a.tournament_k, crossover_rate=a.crossover_rate,
        mutation_rate=a.mutation_rate, problem=a.problem, jump_k=a.jump_k,
        trap_k=a.trap_k, seed=a.seed, output_prefix=a.output_prefix,
    )


if __name__ == "__main__":
    run(parse_args())
