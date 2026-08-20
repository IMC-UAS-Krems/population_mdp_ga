#!/usr/bin/env python3
"""Compare alternative aggregate-state models for a genetic algorithm.

The experiment uses independent training, validation, and test trajectory
splits. Training trajectories estimate aggregate transition matrices, the
validation split selects quantization and time-window hyperparameters, and the
test split provides the final out-of-sample comparison.

Three descriptor families are evaluated:
    full      Fitness, diversity, allele, and selection-aware descriptors.
    fitness   Best-fitness structure, relative fitness counts, mean, variance.
    diversity Allele histograms and diversity-related descriptors only.

The script writes trajectory data, error summaries, a LaTeX table, metadata,
and publication-ready PNG/PDF plots.
"""

from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ExperimentConfig:
    problem: str = "onemax"
    n: int = 20
    mu: int = 16
    generations: int = 100
    train_runs: int = 10000
    validation_runs: int = 2000
    test_runs: int = 10000
    tournament_k: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float | None = None
    jump_k: int = 4
    trap_k: int = 4
    seed: int = 20260818
    candidate_bins: Tuple[int, ...] = (3, 4, 5)
    candidate_windows: Tuple[int, ...] = (0, 10, 20)
    allele_hist_bins: int = 5
    fitness_depth: int = 3
    min_transition_count: int = 1
    output_prefix: str = "aggregate_model_comparison"

    @property
    def pm(self):
        return 1 / self.n if self.mutation_rate is None else self.mutation_rate


@dataclass
class TrajectoryDataset:
    features: np.ndarray
    best: np.ndarray
    diversity: np.ndarray
    target: np.ndarray


@dataclass
class Quantizer:
    edges: Tuple[np.ndarray, ...]
    continuous: Tuple[int, ...]
    exact: Tuple[int, ...]
    window: int

    def encode(self, row, generation, target=0.0):
        e = tuple(int(round(row[i])) for i in self.exact)
        q = tuple(
            int(np.searchsorted(b, row[i], side="right"))
            for b, i in zip(self.edges, self.continuous)
        )
        w = () if self.window <= 0 else (generation // self.window,)
        # Auxiliary flag only for target-compatible absorbing states.
        return e + q + w + (int(target > 0.5),)


@dataclass
class AggregateModel:
    states: list
    index: dict
    K: csr_matrix
    best_readout: np.ndarray
    div_readout: np.ndarray
    target_mask: np.ndarray
    coordinates: np.ndarray


def bit_count(x):
    return int(x.bit_count())


def create_fitness_function(c):
    n = c.n
    if c.problem == "onemax":
        return lambda x: float(bit_count(x)), float(n)
    if c.problem == "leadingones":

        def f(x):
            s = format(x, f"0{n}b")
            return float(len(s) - len(s.lstrip("1")))

        return f, float(n)
    if c.problem == "jump":

        def f(x):
            u = bit_count(x)
            return float(u if u == n or u <= n - c.jump_k else n - u)

        return f, float(n)
    if c.problem == "trap":
        if n % c.trap_k:
            raise ValueError("n must be divisible by trap_k")
        mask = (1 << c.trap_k) - 1

        def f(x):
            z = 0
            for b in range(n // c.trap_k):
                u = bit_count((x >> (b * c.trap_k)) & mask)
                z += c.trap_k if u == c.trap_k else c.trap_k - 1 - u
            return float(z)

        return f, float(n)
    raise ValueError(c.problem)


def tournament(pop, values, k, rng):
    pos = rng.integers(0, len(pop), size=k)
    v = values[pos]
    return int(pop[int(rng.choice(pos[v == v.max()]))])


def tournament_winner_probabilities(values, k):
    mu = len(values)
    out = np.zeros(mu)
    for sample in np.ndindex(*(mu,) * k):
        v = values[list(sample)]
        ws = [j for j, p in enumerate(sample) if values[p] == v.max()]
        mass = (1 / mu) ** k / len(ws)
        for j in ws:
            out[sample[j]] += mass
    return out


def compute_population_descriptors(pop, fitness, c):
    v = np.asarray([fitness(x) for x in pop], float)
    best = float(v.max())
    nmax = int((v == best).sum())
    rel = [int((v == best - d).sum()) for d in range(c.fitness_depth)]
    rel.append(int((v < best - (c.fitness_depth - 1)).sum()))
    p = np.zeros(c.n)
    for x in pop:
        for j in range(c.n):
            p[j] += (x >> j) & 1
    p /= c.mu
    ah, _ = np.histogram(p, bins=np.linspace(0, 1, c.allele_hist_bins + 1))
    counts = Counter(pop)
    ps = 0.0
    gs = list(counts)
    for i, x in enumerate(gs):
        for y in gs[i + 1 :]:
            ps += counts[x] * counts[y] * bit_count(x ^ y)
    div = ps / (c.n * math.comb(c.mu, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(
            (p == 0) | (p == 1), 0, -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        )
    wp = tournament_winner_probabilities(v, c.tournament_k)
    dsel = 0.0
    for i, x in enumerate(pop):
        for j, y in enumerate(pop):
            dsel += wp[i] * wp[j] * bit_count(x ^ y) / c.n
    row = np.asarray(
        [best, nmax, *rel, *ah, v.mean(), v.var(), div, dsel, float(ent.mean())], float
    )
    return row, best, div


def simulate_trajectories(c, runs, seed):
    fit, opt = create_fitness_function(c)
    rng = np.random.default_rng(seed)
    d = len(compute_population_descriptors([0] * c.mu, fit, c)[0])
    F = np.empty((runs, c.generations + 1, d), np.float32)
    B = np.empty((runs, c.generations + 1), np.float32)
    D = np.empty_like(B)
    T = np.empty_like(B)
    for r in range(runs):
        pop = [int(rng.integers(0, 1 << c.n)) for _ in range(c.mu)]
        for t in range(c.generations + 1):
            row, best, div = compute_population_descriptors(pop, fit, c)
            F[r, t] = row
            B[r, t] = best
            D[r, t] = div
            T[r, t] = best >= opt - 1e-12
            if t == c.generations:
                break
            vals = np.asarray([fit(x) for x in pop])
            p1 = tournament(pop, vals, c.tournament_k, rng)
            p2 = tournament(pop, vals, c.tournament_k, rng)
            if rng.random() < c.crossover_rate:
                mask = int(rng.integers(0, 1 << c.n))
                full = (1 << c.n) - 1
                child = (p1 & mask) | (p2 & (~mask & full))
            else:
                child = p1 if rng.random() < 0.5 else p2
            for bit in range(c.n):
                if rng.random() < c.pm:
                    child ^= 1 << bit
            enlarged = pop + [child]
            ev = np.asarray([fit(x) for x in enlarged])
            worst = np.flatnonzero(ev == ev.min())
            del enlarged[int(rng.choice(worst))]
            pop = enlarged
    return TrajectoryDataset(F, B, D, T)


def descriptor_layout(c, name):
    rs = 2
    re = rs + c.fitness_depth + 1
    als = re
    ale = als + c.allele_hist_bins
    mean, var, div, dsel, entropy = range(ale, ale + 5)
    if name == "full":
        return tuple(range(ale)), (mean, var, div, dsel, entropy)
    if name == "fitness":
        return (0, 1, *range(rs, re)), (mean, var)
    if name == "div":
        return tuple(range(als, ale)), (div, dsel, entropy)
    raise ValueError(name)


def fit_quantizer(data, c, name, bins, window):
    exact, cont = descriptor_layout(c, name)
    flat = data.features[:, :, cont].reshape(-1, len(cont))
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return Quantizer(
        tuple(np.unique(np.quantile(flat[:, j], qs)) for j in range(flat.shape[1])),
        cont,
        exact,
        window,
    )


def encode_dataset(data, q):
    return [
        [
            q.encode(data.features[r, t], t, data.target[r, t])
            for t in range(data.features.shape[1])
        ]
        for r in range(data.features.shape[0])
    ]


def fit_aggregate_model(data, q, c):
    E = encode_dataset(data, q)
    states = sorted({z for tr in E for z in tr})
    ix = {z: i for i, z in enumerate(states)}
    trans = defaultdict(Counter)
    bs = np.zeros(len(states))
    ds = np.zeros(len(states))
    visits = np.zeros(len(states))
    target = np.zeros(len(states), bool)
    for r, tr in enumerate(E):
        for t, z in enumerate(tr):
            i = ix[z]
            visits[i] += 1
            bs[i] += data.best[r, t]
            ds[i] += data.diversity[r, t]
            target[i] |= bool(data.target[r, t])
            if t < c.generations:
                trans[i][ix[tr[t + 1]]] += 1
    rows = []
    cols = []
    vals = []
    for i in range(len(states)):
        if target[i] or not trans[i]:
            rows.append(i)
            cols.append(i)
            vals.append(1.0)
            continue
        total = sum(trans[i].values())
        used = 0
        for j, n in trans[i].items():
            if n >= c.min_transition_count:
                rows.append(i)
                cols.append(j)
                vals.append(n / total)
                used += n
        if used < total:
            rows.append(i)
            cols.append(i)
            vals.append((total - used) / total)
    K = csr_matrix((vals, (rows, cols)), shape=(len(states), len(states)))
    return AggregateModel(
        states, ix, K, bs / visits, ds / visits, target, np.asarray(states, float)
    )


def nearest_aggregate_state(z, m):
    q = np.asarray(z, float)
    scale = np.maximum(1, np.ptp(m.coordinates, axis=0))
    return int(np.argmin(np.sum(abs(m.coordinates - q) / scale, axis=1)))


def estimate_initial_distribution(data, q, m):
    d = np.zeros(len(m.states))
    unseen = 0
    for r in range(data.features.shape[0]):
        z = q.encode(data.features[r, 0], 0, data.target[r, 0])
        i = m.index.get(z)
        if i is None:
            unseen += 1
            i = nearest_aggregate_state(z, m)
        d[i] += 1
    return d / d.sum(), unseen / data.features.shape[0]


def predict_aggregate_trajectories(d, m, T):
    out = {k: np.empty(T + 1) for k in ("best", "diversity", "success")}
    target = m.target_mask.astype(float)
    for t in range(T + 1):
        out["best"][t] = d @ m.best_readout
        out["diversity"][t] = d @ m.div_readout
        out["success"][t] = d @ target
        d = d @ m.K
    out["success"] = np.maximum.accumulate(out["success"])
    return out


def empirical_trajectories(data):
    return {
        "best": data.best.mean(0),
        "diversity": data.diversity.mean(0),
        "success": np.maximum.accumulate(data.target.mean(0)),
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_prediction_scores(pred, obs, c):
    b = rmse(pred["best"], obs["best"]) / c.n
    d = rmse(pred["diversity"], obs["diversity"])
    s = rmse(pred["success"], obs["success"])
    return {
        "total_score": b + d + s,
        "best_nrmse": b,
        "diversity_rmse": d,
        "success_rmse": s,
    }


def write_latex_table(frame, path):
    labels = {"full": "Full rich", "fitness": "Fitness", "div": "Diversity"}
    L = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \caption{Predictive accuracy of the aggregate-state models on independent test trajectories. The total score is the sum of the normalized best-fitness RMSE, diversity RMSE, and success-probability RMSE.}",
        r"    \label{tab:aggregate-model-comparison}",
        r"    \begin{tabular}{lrrrr}",
        r"        \hline",
        r"        \textbf{Model} & \textbf{Total score} & \textbf{Best NRMSE} & \textbf{Diversity RMSE} & \textbf{Success RMSE} \\",
        r"        \hline",
    ]
    for _, x in frame.iterrows():
        L.append(
            "        {} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\".format(
                labels[x.model],
                x.total_score,
                x.best_nrmse,
                x.diversity_rmse,
                x.success_rmse,
            )
        )
    L += [r"        \hline", r"    \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(L))


def run_experiment(c):
    print("Simulating independent splits...")
    train = simulate_trajectories(c, c.train_runs, c.seed + 1)
    valid = simulate_trajectories(c, c.validation_runs, c.seed + 2)
    test = simulate_trajectories(c, c.test_runs, c.seed + 3)
    ov = empirical_trajectories(valid)
    ot = empirical_trajectories(test)
    selected = {}
    for name in ("full", "fitness", "div"):
        cand = []
        for bins in c.candidate_bins:
            for window in c.candidate_windows:
                q = fit_quantizer(train, c, name, bins, window)
                m = fit_aggregate_model(train, q, c)
                d, u = estimate_initial_distribution(valid, q, m)
                pr = predict_aggregate_trajectories(d, m, c.generations)
                sc = compute_prediction_scores(pr, ov, c)["total_score"]
                cand.append((sc, bins, window, q, m, u))
                print(
                    f"{name:8s} bins={bins} window={window:2d} states={len(m.states):6d} score={sc:.6f} unseen={u:.3f}"
                )
        sc, bins, window, q, m, _ = min(cand, key=lambda x: x[0])
        d, u = estimate_initial_distribution(test, q, m)
        selected[name] = {
            "validation_score": sc,
            "bins": bins,
            "window": window,
            "model": m,
            "prediction": predict_aggregate_trajectories(d, m, c.generations),
            "unseen": u,
        }
    prefix = Path(c.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    traj = []
    score_rows = []
    metric_rows = []
    for name, x in selected.items():
        pr = x["prediction"]
        score_rows.append({"model": name, **compute_prediction_scores(pr, ot, c)})
        for target in pr:
            metric_rows.append(
                {
                    "model": name,
                    "target": target,
                    "bins": x["bins"],
                    "window": x["window"],
                    "states": len(x["model"].states),
                    "unseen": x["unseen"],
                    "rmse": rmse(pr[target], ot[target]),
                }
            )
        for t in range(c.generations + 1):
            traj.append(
                {
                    "model": name,
                    "generation": t,
                    **{f"predicted_{k}": pr[k][t] for k in pr},
                    **{f"observed_{k}": ot[k][t] for k in ot},
                }
            )
    sf = pd.DataFrame(score_rows).sort_values("total_score")
    pd.DataFrame(traj).to_csv(
        prefix.with_name(prefix.name + "_trajectories").with_suffix(".csv"), index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        prefix.with_name(prefix.name + "_metrics").with_suffix(".csv"), index=False
    )
    sf.to_csv(
        prefix.with_name(prefix.name + "_scores").with_suffix(".csv"), index=False
    )
    write_latex_table(sf, prefix.with_name(prefix.name + "_scores").with_suffix(".tex"))
    meta = {
        "config": {**asdict(c), "effective_mutation_rate": c.pm},
        "target_flag_note": "An auxiliary target flag is appended only to make aggregate target states absorbing; it is not a predictive descriptor.",
        "selected": {
            n: {
                "bins": x["bins"],
                "window": x["window"],
                "validation_score": x["validation_score"],
                "states": len(x["model"].states),
                "test_unseen": x["unseen"],
            }
            for n, x in selected.items()
        },
    }
    prefix.with_name(prefix.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(meta, indent=2)
    )
    labels = {"full": "Full rich", "fitness": "Fitness", "div": "Diversity"}
    colors = {"full": "tab:red", "fitness": "tab:blue", "div": "tab:green"}
    xx = np.arange(c.generations + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, k, yl in zip(
        axs,
        ("best", "diversity", "success"),
        (
            "Expected best fitness",
            "Expected normalized diversity",
            "Success probability",
        ),
    ):
        ax.plot(xx, ot[k], color="black", lw=2.5, label="Independent test simulation")
        for n in ("full", "fitness", "div"):
            ax.plot(
                xx, selected[n]["prediction"][k], color=colors[n], lw=2, label=labels[n]
            )
        ax.set_xlabel("Generation")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.suptitle(
        f"{c.problem}: aggregate-model comparison, n={c.n}, mu={c.mu}, pc={c.crossover_rate}"
    )
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(sf.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--problem", choices=["onemax", "leadingones", "jump", "trap"], default="onemax"
    )
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--mu", type=int, default=16)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--train-runs", type=int, default=10000)
    p.add_argument("--validation-runs", type=int, default=2000)
    p.add_argument("--test-runs", type=int, default=10000)
    p.add_argument("--tournament-k", type=int, default=2)
    p.add_argument("--crossover-rate", type=float, default=0.9)
    p.add_argument("--mutation-rate", type=float, default=None)
    p.add_argument("--jump-k", type=int, default=4)
    p.add_argument("--trap-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--candidate-bins", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--candidate-windows", type=int, nargs="+", default=[0, 10, 20])
    p.add_argument("--allele-hist-bins", type=int, default=5)
    p.add_argument("--fitness-depth", type=int, default=3)
    p.add_argument("--min-transition-count", type=int, default=1)
    p.add_argument("--output-prefix", default="aggregate_model_comparison")
    a = p.parse_args()
    return ExperimentConfig(
        problem=a.problem,
        n=a.n,
        mu=a.mu,
        generations=a.generations,
        train_runs=a.train_runs,
        validation_runs=a.validation_runs,
        test_runs=a.test_runs,
        tournament_k=a.tournament_k,
        crossover_rate=a.crossover_rate,
        mutation_rate=a.mutation_rate,
        jump_k=a.jump_k,
        trap_k=a.trap_k,
        seed=a.seed,
        candidate_bins=tuple(a.candidate_bins),
        candidate_windows=tuple(a.candidate_windows),
        allele_hist_bins=a.allele_hist_bins,
        fitness_depth=a.fitness_depth,
        min_transition_count=a.min_transition_count,
        output_prefix=a.output_prefix,
    )


if __name__ == "__main__":
    run_experiment(parse_arguments())
