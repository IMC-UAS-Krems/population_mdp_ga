#!/usr/bin/env python3
"""Estimate the finite-horizon value of diversity and crossover in a GA.

The experiment constructs pairs of binary populations that are closely matched
on fitness descriptors but differ in normalized pairwise Hamming diversity.
Each population is evaluated under two otherwise identical GA policies:

    pi_cross : uniform crossover enabled
    pi_mut   : mutation-only search (parental copying instead of crossover)

For a terminal performance functional Phi and horizon T, the script estimates

    VoD_cross = V_cross(P_high) - V_cross(P_low)
    VoD_mut   = V_mut(P_high)   - V_mut(P_low)

and the diversity-crossover interaction

    Synergy = [V_cross(P_high) - V_mut(P_high)]
              - [V_cross(P_low) - V_mut(P_low)]
            = VoD_cross - VoD_mut.

A positive interaction indicates that crossover obtains more future value from
the high-diversity population than from the matched low-diversity population.

The default performance targets are:
    * expected best fitness at horizons in --horizons;
    * probability of finding an optimum by those horizons;
    * expected normalized diversity at those horizons.

Pair construction is separated from rollout evaluation. Candidate populations
are generated first, grouped by exact/coarsened fitness descriptors, and paired
within each group using the largest available diversity contrast. Rollout seeds
are paired across the four experimental cells to reduce Monte Carlo noise.

Example
-------
python diversity_crossover_interaction.py \
    --problem jump --n 20 --mu 16 --jump-k 4 \
    --candidate-populations 20000 --matched-pairs 100 \
    --rollouts 1000 --horizons 25 50 100 \
    --output-prefix jump_diversity_crossover
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Population = Tuple[int, ...]
FitnessFunction = Callable[[int], float]


@dataclass(frozen=True)
class ExperimentConfig:
    """Algorithm, matching, simulation, and output settings."""

    problem: str = "jump"
    n: int = 20
    mu: int = 16
    jump_k: int = 4
    trap_k: int = 4
    tournament_k: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    candidate_populations: int = 20_000
    matched_pairs: int = 100
    rollouts: int = 1_000
    horizons: Tuple[int, ...] = (25, 50, 100)
    mean_fitness_bin_width: float = 0.25
    variance_bin_width: float = 0.25
    minimum_diversity_gap: float = 0.10
    minimum_group_size: int = 2
    seed: int = 20260820
    bootstrap_samples: int = 2_000
    output_prefix: str = "diversity_crossover_interaction"

    @property
    def bit_mutation_rate(self) -> float:
        return 1.0 / self.n if self.mutation_rate is None else self.mutation_rate

    @property
    def maximum_horizon(self) -> int:
        return max(self.horizons)


@dataclass(frozen=True)
class PopulationSummary:
    """A candidate population and the descriptors used for matching."""

    population: Population
    best_fitness: float
    mean_fitness: float
    fitness_variance: float
    best_count: int
    diversity: float


@dataclass(frozen=True)
class MatchedPair:
    """Fitness-matched populations with different diversity values."""

    pair_id: int
    low: PopulationSummary
    high: PopulationSummary

    @property
    def diversity_gap(self) -> float:
        return self.high.diversity - self.low.diversity


@dataclass
class RolloutResult:
    """Mean rollout trajectories for one population-policy cell."""

    best_fitness: np.ndarray
    diversity: np.ndarray
    success: np.ndarray


def bit_count(genotype: int) -> int:
    return int(genotype.bit_count())


def create_fitness_function(config: ExperimentConfig) -> Tuple[FitnessFunction, float]:
    """Return the chosen pseudo-Boolean objective and its optimum value."""

    n = config.n
    if config.problem == "onemax":
        return lambda x: float(bit_count(x)), float(n)

    if config.problem == "leadingones":
        def leading_ones(x: int) -> float:
            bits = format(x, f"0{n}b")
            return float(len(bits) - len(bits.lstrip("1")))
        return leading_ones, float(n)

    if config.problem == "jump":
        k = config.jump_k
        if not 1 <= k < n:
            raise ValueError("jump_k must satisfy 1 <= jump_k < n")

        def jump(x: int) -> float:
            ones = bit_count(x)
            if ones == n or ones <= n - k:
                return float(ones)
            return float(n - ones)

        return jump, float(n)

    if config.problem == "trap":
        k = config.trap_k
        if n % k:
            raise ValueError("n must be divisible by trap_k")
        mask = (1 << k) - 1

        def concatenated_trap(x: int) -> float:
            value = 0.0
            for block in range(n // k):
                ones = bit_count((x >> (block * k)) & mask)
                value += k if ones == k else k - 1 - ones
            return value

        return concatenated_trap, float(n)

    raise ValueError(f"Unsupported problem: {config.problem}")


def normalized_pairwise_diversity(population: Sequence[int], n: int) -> float:
    """Normalized mean pairwise Hamming distance in [0, 1]."""

    mu = len(population)
    if mu < 2:
        return 0.0
    counts = Counter(population)
    total_distance = 0.0
    genotypes = list(counts)
    for i, first in enumerate(genotypes):
        for second in genotypes[i + 1:]:
            total_distance += (
                counts[first] * counts[second] * bit_count(first ^ second)
            )
    return total_distance / (n * math.comb(mu, 2))


def summarize_population(
    population: Population,
    fitness: FitnessFunction,
    config: ExperimentConfig,
) -> PopulationSummary:
    values = np.fromiter((fitness(x) for x in population), dtype=float)
    best = float(values.max())
    return PopulationSummary(
        population=population,
        best_fitness=best,
        mean_fitness=float(values.mean()),
        fitness_variance=float(values.var()),
        best_count=int(np.sum(values == best)),
        diversity=normalized_pairwise_diversity(population, config.n),
    )


def quantize(value: float, width: float) -> int:
    """Assign a continuous descriptor to a fixed-width matching bin."""

    if width <= 0:
        raise ValueError("Matching bin widths must be positive")
    return int(math.floor(value / width + 1e-12))


def matching_key(summary: PopulationSummary, config: ExperimentConfig) -> Tuple[int, ...]:
    """Fitness descriptors required to agree within a matched group."""

    return (
        int(round(summary.best_fitness)),
        summary.best_count,
        quantize(summary.mean_fitness, config.mean_fitness_bin_width),
        quantize(summary.fitness_variance, config.variance_bin_width),
    )


def generate_candidate_populations(
    config: ExperimentConfig,
    fitness: FitnessFunction,
) -> List[PopulationSummary]:
    """Sample candidate populations independently from the uniform initializer."""

    rng = np.random.default_rng(config.seed)
    candidates: List[PopulationSummary] = []
    for _ in range(config.candidate_populations):
        population = tuple(
            int(x) for x in rng.integers(0, 1 << config.n, size=config.mu)
        )
        candidates.append(summarize_population(population, fitness, config))
    return candidates


def genotypes_by_fitness_level(
    config: ExperimentConfig,
    fitness: FitnessFunction,
) -> dict[int, list[int]]:
    """Enumerate genotypes and group them by integer fitness.

    This is feasible for n=20 because the binary search space contains
    2**20 = 1,048,576 genotypes. For substantially larger n, genotype
    pools should instead be generated by constructive or rejection sampling.
    """
    if config.n > 22:
        raise ValueError(
            "Exact level enumeration is intended for n <= 22. "
            "Use level-specific sampling for larger dimensions."
        )

    levels: dict[int, list[int]] = defaultdict(list)
    for genotype in range(1 << config.n):
        level = int(round(fitness(genotype)))
        levels[level].append(genotype)

    return dict(levels)

def sample_fitness_profile(
    available_levels: Sequence[int],
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Sample the common individual-fitness vector for a matched pair.

    Profiles are biased toward relevant high-fitness regions because the
    future value of diversity and crossover is most informative there.
    """
    levels = np.asarray(sorted(available_levels), dtype=int)

    if config.problem == "jump":
        local_optimum_level = config.n - config.jump_k

        admissible = levels[
            (levels >= max(0, local_optimum_level - 3))
            & (levels <= local_optimum_level)
        ]
    else:
        admissible = levels[levels >= max(0, config.n // 2)]

    if len(admissible) == 0:
        admissible = levels

    # Ensure that several individuals attain the best sampled level.
    best_level = int(rng.choice(admissible))
    best_count = int(rng.integers(2, max(3, config.mu // 2 + 1)))

    lower_levels = admissible[admissible <= best_level]
    remainder = rng.choice(
        lower_levels,
        size=config.mu - best_count,
        replace=True,
    )

    profile = np.concatenate(
        [
            np.full(best_count, best_level, dtype=int),
            remainder.astype(int),
        ]
    )
    rng.shuffle(profile)
    return tuple(int(level) for level in profile)

def minimum_distance_to_population(
    candidate: int,
    selected: Sequence[int],
    n: int,
) -> float:
    """Minimum normalized Hamming distance to already selected genotypes."""
    if not selected:
        return 1.0

    return min(
        bit_count(candidate ^ existing) / n
        for existing in selected
    )

def construct_low_diversity_population(
    fitness_profile: Sequence[int],
    level_pools: dict[int, list[int]],
    rng: np.random.Generator,
) -> Population:
    """Construct a population with the requested fitness values and low diversity.

    One representative genotype is selected for each fitness level and reused
    whenever that level occurs repeatedly. This creates genotype duplicates
    while preserving the complete individual-fitness profile.
    """
    representative_by_level: dict[int, int] = {}
    population: list[int] = []

    for level in fitness_profile:
        if level not in representative_by_level:
            representative_by_level[level] = int(
                rng.choice(level_pools[level])
            )

        population.append(representative_by_level[level])

    return tuple(population)

def construct_high_diversity_population(
    fitness_profile: Sequence[int],
    level_pools: dict[int, list[int]],
    config: ExperimentConfig,
    rng: np.random.Generator,
    candidate_pool_size: int = 256,
) -> Population:
    """Construct a population with the same fitness profile and high diversity.

    For each required fitness level, candidate genotypes are sampled from the
    corresponding level set. The genotype maximizing its minimum Hamming
    distance to the individuals already selected is retained.
    """
    population: list[int] = []

    # Process repeated and difficult levels first.
    level_counts = Counter(fitness_profile)
    ordered_levels = sorted(
        fitness_profile,
        key=lambda level: (-level_counts[level], -level),
    )

    for level in ordered_levels:
        pool = level_pools[level]

        if len(pool) <= candidate_pool_size:
            candidates = pool
        else:
            candidate_indices = rng.choice(
                len(pool),
                size=candidate_pool_size,
                replace=False,
            )
            candidates = [pool[int(index)] for index in candidate_indices]

        # Prefer unused genotypes if the level set is sufficiently large.
        unused = [
            candidate for candidate in candidates
            if candidate not in population
        ]
        if unused:
            candidates = unused

        selected = max(
            candidates,
            key=lambda candidate: minimum_distance_to_population(
                candidate,
                population,
                config.n,
            ),
        )
        population.append(int(selected))

    # Ordering has no meaning for a multiset population.
    return tuple(population)

def construct_controlled_matched_pairs(
    config: ExperimentConfig,
    fitness: FitnessFunction,
) -> list[MatchedPair]:
    """Construct fitness-identical population pairs with contrasting diversity."""
    rng = np.random.default_rng(config.seed)
    level_pools = genotypes_by_fitness_level(config, fitness)
    available_levels = sorted(level_pools)

    pairs: list[MatchedPair] = []
    attempts = 0
    maximum_attempts = max(
        10_000,
        500 * config.matched_pairs,
    )

    while (
        len(pairs) < config.matched_pairs
        and attempts < maximum_attempts
    ):
        attempts += 1

        profile = sample_fitness_profile(
            available_levels,
            config,
            rng,
        )

        # Skip profiles containing levels with no genotype pool.
        if any(level not in level_pools for level in profile):
            continue

        low_population = construct_low_diversity_population(
            profile,
            level_pools,
            rng,
        )
        high_population = construct_high_diversity_population(
            profile,
            level_pools,
            config,
            rng,
        )

        low_summary = summarize_population(
            low_population,
            fitness,
            config,
        )
        high_summary = summarize_population(
            high_population,
            fitness,
            config,
        )

        # Reorder in the unlikely case that the greedy construction does not
        # produce the intended direction.
        if high_summary.diversity < low_summary.diversity:
            low_summary, high_summary = high_summary, low_summary

        diversity_gap = (
            high_summary.diversity
            - low_summary.diversity
        )

        if diversity_gap < config.minimum_diversity_gap:
            continue

        # These assertions verify exact fitness matching.
        np.testing.assert_allclose(
            low_summary.best_fitness,
            high_summary.best_fitness,
        )
        np.testing.assert_allclose(
            low_summary.mean_fitness,
            high_summary.mean_fitness,
        )
        np.testing.assert_allclose(
            low_summary.fitness_variance,
            high_summary.fitness_variance,
        )
        assert low_summary.best_count == high_summary.best_count

        pairs.append(
            MatchedPair(
                pair_id=len(pairs),
                low=low_summary,
                high=high_summary,
            )
        )

    if len(pairs) < config.matched_pairs:
        observed_message = (
            f"Constructed only {len(pairs)} of the requested "
            f"{config.matched_pairs} matched pairs after "
            f"{attempts} attempts. "
        )
        if pairs:
            mean_gap = np.mean(
                [pair.diversity_gap for pair in pairs]
            )
            observed_message += (
                f"The mean accepted diversity gap was {mean_gap:.4f}. "
            )
        observed_message += (
            "Reduce --minimum-diversity-gap or increase the number of "
            "high-diversity candidates considered per fitness level."
        )
        raise RuntimeError(observed_message)

    return pairs


def construct_matched_pairs(
    candidates: Iterable[PopulationSummary],
    config: ExperimentConfig,
) -> List[MatchedPair]:
    """Match on fitness descriptors and maximize diversity contrast per group.

    Each candidate is used in at most one pair. Within every matching group,
    populations are sorted by diversity and paired from the two extremes.
    """

    groups: Dict[Tuple[int, ...], List[PopulationSummary]] = defaultdict(list)
    for candidate in candidates:
        groups[matching_key(candidate, config)].append(candidate)

    provisional: List[Tuple[float, PopulationSummary, PopulationSummary]] = []
    for group in groups.values():
        if len(group) < config.minimum_group_size:
            continue
        ordered = sorted(group, key=lambda item: item.diversity)
        left, right = 0, len(ordered) - 1
        while left < right:
            low, high = ordered[left], ordered[right]
            gap = high.diversity - low.diversity
            if gap >= config.minimum_diversity_gap:
                provisional.append((gap, low, high))
            left += 1
            right -= 1

    provisional.sort(key=lambda item: item[0], reverse=True)
    selected = provisional[: config.matched_pairs]
    if not selected:
        raise RuntimeError(
            "No matched pairs were found. Increase candidate-populations, "
            "widen the fitness bins, or reduce minimum-diversity-gap."
        )

    return [
        MatchedPair(pair_id=i, low=low, high=high)
        for i, (_, low, high) in enumerate(selected)
    ]


def tournament_select(
    population: Sequence[int],
    fitness_values: np.ndarray,
    tournament_k: int,
    rng: np.random.Generator,
) -> int:
    positions = rng.integers(0, len(population), size=tournament_k)
    sampled_values = fitness_values[positions]
    winners = positions[sampled_values == sampled_values.max()]
    return int(population[int(rng.choice(winners))])


def produce_child(
    population: Sequence[int],
    fitness: FitnessFunction,
    crossover_rate: float,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> int:
    """Apply tournament selection, optional uniform crossover, and mutation."""

    values = np.fromiter((fitness(x) for x in population), dtype=float)
    parent_1 = tournament_select(population, values, config.tournament_k, rng)
    parent_2 = tournament_select(population, values, config.tournament_k, rng)

    if rng.random() < crossover_rate:
        mask = int(rng.integers(0, 1 << config.n))
        full_mask = (1 << config.n) - 1
        child = (parent_1 & mask) | (parent_2 & (~mask & full_mask))
    else:
        child = parent_1 if rng.random() < 0.5 else parent_2

    for bit in range(config.n):
        if rng.random() < config.bit_mutation_rate:
            child ^= 1 << bit
    return child


def elitist_replacement(
    population: Sequence[int],
    child: int,
    fitness: FitnessFunction,
    rng: np.random.Generator,
) -> List[int]:
    """Insert one child and uniformly delete one minimum-fitness member."""

    enlarged = list(population) + [child]
    values = np.fromiter((fitness(x) for x in enlarged), dtype=float)
    worst_positions = np.flatnonzero(values == values.min())
    del enlarged[int(rng.choice(worst_positions))]
    return enlarged


def rollout_population(
    initial_population: Population,
    fitness: FitnessFunction,
    optimum: float,
    crossover_rate: float,
    rollout_seeds: np.ndarray,
    config: ExperimentConfig,
) -> RolloutResult:
    """Estimate mean trajectories from one initial population and policy.

    The supplied seeds are reused across matched experimental cells. This
    common-random-number design reduces variance in pairwise contrasts.
    """

    horizon = config.maximum_horizon
    best = np.empty((config.rollouts, horizon + 1), dtype=np.float32)
    diversity = np.empty_like(best)
    success = np.empty_like(best)

    for run, seed in enumerate(rollout_seeds):
        rng = np.random.default_rng(int(seed))
        population = list(initial_population)
        found_optimum = False

        for generation in range(horizon + 1):
            values = np.fromiter((fitness(x) for x in population), dtype=float)
            current_best = float(values.max())
            found_optimum = found_optimum or current_best >= optimum - 1e-12
            best[run, generation] = current_best
            diversity[run, generation] = normalized_pairwise_diversity(
                population, config.n
            )
            success[run, generation] = float(found_optimum)

            if generation == horizon:
                break
            child = produce_child(
                population, fitness, crossover_rate, config, rng
            )
            population = elitist_replacement(population, child, fitness, rng)

    return RolloutResult(
        best_fitness=best.mean(axis=0),
        diversity=diversity.mean(axis=0),
        success=success.mean(axis=0),
    )


def bootstrap_mean_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Return mean and percentile bootstrap 95% confidence interval."""

    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def evaluate_pairs(
    pairs: Sequence[MatchedPair],
    fitness: FitnessFunction,
    optimum: float,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 2x2 diversity-by-crossover experiment for every pair."""

    rng = np.random.default_rng(config.seed + 10_000)
    pair_rows: List[dict] = []
    trajectory_rows: List[dict] = []

    for pair in pairs:
        rollout_seeds = rng.integers(
            0, np.iinfo(np.uint32).max, size=config.rollouts, dtype=np.uint32
        )
        cells = {
            ("low", "mutation"): rollout_population(
                pair.low.population, fitness, optimum, 0.0, rollout_seeds, config
            ),
            ("low", "crossover"): rollout_population(
                pair.low.population,
                fitness,
                optimum,
                config.crossover_rate,
                rollout_seeds,
                config,
            ),
            ("high", "mutation"): rollout_population(
                pair.high.population, fitness, optimum, 0.0, rollout_seeds, config
            ),
            ("high", "crossover"): rollout_population(
                pair.high.population,
                fitness,
                optimum,
                config.crossover_rate,
                rollout_seeds,
                config,
            ),
        }

        for diversity_level, policy in cells:
            result = cells[(diversity_level, policy)]
            for generation in range(config.maximum_horizon + 1):
                trajectory_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "diversity_level": diversity_level,
                        "policy": policy,
                        "generation": generation,
                        "expected_best": result.best_fitness[generation],
                        "expected_diversity": result.diversity[generation],
                        "success_probability": result.success[generation],
                    }
                )

        for horizon in config.horizons:
            for target, attribute in (
                ("best_fitness", "best_fitness"),
                ("success_probability", "success"),
                ("population_diversity", "diversity"),
            ):
                low_mut = float(getattr(cells[("low", "mutation")], attribute)[horizon])
                low_cross = float(
                    getattr(cells[("low", "crossover")], attribute)[horizon]
                )
                high_mut = float(
                    getattr(cells[("high", "mutation")], attribute)[horizon]
                )
                high_cross = float(
                    getattr(cells[("high", "crossover")], attribute)[horizon]
                )

                value_of_diversity_mutation = high_mut - low_mut
                value_of_diversity_crossover = high_cross - low_cross
                crossover_low = low_cross - low_mut
                crossover_high = high_cross - high_mut
                interaction = crossover_high - crossover_low

                pair_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "horizon": horizon,
                        "target": target,
                        "low_initial_best": pair.low.best_fitness,
                        "high_initial_best": pair.high.best_fitness,
                        "low_initial_mean": pair.low.mean_fitness,
                        "high_initial_mean": pair.high.mean_fitness,
                        "low_initial_variance": pair.low.fitness_variance,
                        "high_initial_variance": pair.high.fitness_variance,
                        "low_initial_best_count": pair.low.best_count,
                        "high_initial_best_count": pair.high.best_count,
                        "low_initial_diversity": pair.low.diversity,
                        "high_initial_diversity": pair.high.diversity,
                        "diversity_gap": pair.diversity_gap,
                        "low_mutation": low_mut,
                        "low_crossover": low_cross,
                        "high_mutation": high_mut,
                        "high_crossover": high_cross,
                        "value_of_diversity_mutation": value_of_diversity_mutation,
                        "value_of_diversity_crossover": value_of_diversity_crossover,
                        "crossover_contribution_low": crossover_low,
                        "crossover_contribution_high": crossover_high,
                        "diversity_crossover_interaction": interaction,
                    }
                )

    return pd.DataFrame(pair_rows), pd.DataFrame(trajectory_rows)


def summarize_interactions(
    pair_results: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Aggregate pair-level effects and calculate bootstrap intervals."""

    rows: List[dict] = []
    effects = (
        "value_of_diversity_mutation",
        "value_of_diversity_crossover",
        "crossover_contribution_low",
        "crossover_contribution_high",
        "diversity_crossover_interaction",
    )
    for (horizon, target), group in pair_results.groupby(["horizon", "target"]):
        for effect in effects:
            estimate, lower, upper = bootstrap_mean_interval(
                group[effect].to_numpy(),
                config.bootstrap_samples,
                config.seed + int(horizon) + sum(map(ord, effect)),
            )
            rows.append(
                {
                    "horizon": int(horizon),
                    "target": target,
                    "effect": effect,
                    "mean": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "pairs": len(group),
                    "mean_diversity_gap": float(group["diversity_gap"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_figure(summary: pd.DataFrame, config: ExperimentConfig) -> None:
    """Plot interaction estimates for all targets and horizons."""

    interaction = summary[
        summary["effect"] == "diversity_crossover_interaction"
    ].copy()
    targets = [
        ("best_fitness", "Best-fitness interaction"),
        ("success_probability", "Success-probability interaction"),
        ("population_diversity", "Diversity interaction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    for ax, (target, ylabel) in zip(axes, targets):
        data = interaction[interaction["target"] == target].sort_values("horizon")
        x = data["horizon"].to_numpy()
        y = data["mean"].to_numpy()
        lower = y - data["ci95_lower"].to_numpy()
        upper = data["ci95_upper"].to_numpy() - y
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
            linewidth=2,
            color="tab:purple",
        )
        ax.set_xlabel("Horizon")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{config.problem}: diversity-crossover interaction, "
        f"n={config.n}, mu={config.mu}, pc={config.crossover_rate}"
    )
    fig.tight_layout()
    prefix = Path(config.output_prefix)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    pairs: Sequence[MatchedPair],
    pair_results: pd.DataFrame,
    trajectories: pd.DataFrame,
    summary: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Write all reproducibility, diagnostic, and publication outputs."""

    prefix = Path(config.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pair_table = pd.DataFrame(
        [
            {
                "pair_id": pair.pair_id,
                "low_best": pair.low.best_fitness,
                "high_best": pair.high.best_fitness,
                "low_mean": pair.low.mean_fitness,
                "high_mean": pair.high.mean_fitness,
                "low_variance": pair.low.fitness_variance,
                "high_variance": pair.high.fitness_variance,
                "low_best_count": pair.low.best_count,
                "high_best_count": pair.high.best_count,
                "low_diversity": pair.low.diversity,
                "high_diversity": pair.high.diversity,
                "diversity_gap": pair.diversity_gap,
            }
            for pair in pairs
        ]
    )

    pair_table.to_csv(
        prefix.with_name(prefix.name + "_matched_pairs").with_suffix(".csv"),
        index=False,
    )
    pair_results.to_csv(
        prefix.with_name(prefix.name + "_pair_effects").with_suffix(".csv"),
        index=False,
    )
    trajectories.to_csv(
        prefix.with_name(prefix.name + "_trajectories").with_suffix(".csv"),
        index=False,
    )
    summary.to_csv(
        prefix.with_name(prefix.name + "_summary").with_suffix(".csv"),
        index=False,
    )

    metadata = {
        "config": {
            **asdict(config),
            "effective_mutation_rate": config.bit_mutation_rate,
        },
        "matched_pairs_found": len(pairs),
        "matching_definition": {
            "exact": ["best_fitness", "best_count"],
            "binned": {
                "mean_fitness_width": config.mean_fitness_bin_width,
                "fitness_variance_width": config.variance_bin_width,
            },
            "minimum_diversity_gap": config.minimum_diversity_gap,
        },
        "interaction_definition": (
            "[V_cross(P_high)-V_mut(P_high)] - "
            "[V_cross(P_low)-V_mut(P_low)]"
        ),
    }
    prefix.with_name(prefix.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_figure(summary, config)


def run_experiment(config: ExperimentConfig) -> None:
    """Generate pairs, evaluate the factorial design, and save results."""

    fitness, optimum = create_fitness_function(config)
    print(
        "Constructing fitness-identical high-/low-diversity "
        "population pairs..."
    )
    pairs = construct_controlled_matched_pairs(
        config,
        fitness,
    )
    print(
        f"Selected {len(pairs)} pairs; mean diversity gap = "
        f"{np.mean([p.diversity_gap for p in pairs]):.4f}"
    )

    print("Evaluating crossover and mutation-only policies...")
    pair_results, trajectories = evaluate_pairs(pairs, fitness, optimum, config)
    summary = summarize_interactions(pair_results, config)
    save_outputs(pairs, pair_results, trajectories, summary, config)

    print("\nDiversity-crossover interaction estimates:")
    columns = ["horizon", "target", "mean", "ci95_lower", "ci95_upper", "pairs"]
    interaction = summary[
        summary["effect"] == "diversity_crossover_interaction"
    ][columns]
    print(interaction.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOutputs written with prefix: {config.output_prefix}")


def parse_arguments() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        choices=["onemax", "leadingones", "jump", "trap"],
        default="jump",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--mu", type=int, default=16)
    parser.add_argument("--jump-k", type=int, default=4)
    parser.add_argument("--trap-k", type=int, default=4)
    parser.add_argument("--tournament-k", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--candidate-populations", type=int, default=20_000)
    parser.add_argument("--matched-pairs", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=1_000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--mean-fitness-bin-width", type=float, default=0.25)
    parser.add_argument("--variance-bin-width", type=float, default=0.25)
    parser.add_argument("--minimum-diversity-gap", type=float, default=0.10)
    parser.add_argument("--minimum-group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--output-prefix", default="diversity_crossover_interaction"
    )
    args = parser.parse_args()

    return ExperimentConfig(
        problem=args.problem,
        n=args.n,
        mu=args.mu,
        jump_k=args.jump_k,
        trap_k=args.trap_k,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        candidate_populations=args.candidate_populations,
        matched_pairs=args.matched_pairs,
        rollouts=args.rollouts,
        horizons=tuple(sorted(set(args.horizons))),
        mean_fitness_bin_width=args.mean_fitness_bin_width,
        variance_bin_width=args.variance_bin_width,
        minimum_diversity_gap=args.minimum_diversity_gap,
        minimum_group_size=args.minimum_group_size,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    run_experiment(parse_arguments())
