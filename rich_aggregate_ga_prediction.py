#!/usr/bin/env python3
"""Rich aggregate-state prediction for a steady-state genetic algorithm.

This experiment addresses failures of coarse aggregate states by adding:
  * exact best fitness (never quantized),
  * number of best-fitness individuals,
  * a relative fitness histogram,
  * a histogram of locus-level allele frequencies,
  * normalized pairwise Hamming diversity,
  * allele entropy,
  * expected selected-parent distance,
  * an optional generation-window variable.

Success probabilities are computed with target-compatible absorbing aggregate
states, not from a historical-success state readout.

Outputs:
  PREFIX.png / PREFIX.pdf
  PREFIX_trajectories.csv
  PREFIX_metrics.csv
  PREFIX_metadata.json

Example:
  python rich_aggregate_ga_prediction.py --problem onemax --n 20 --mu 16 \
      --generations 100 --train-runs 10000 --validation-runs 2000 \
      --test-runs 10000 --output-prefix rich_onemax
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Config:
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
    output_prefix: str = "rich_aggregate_ga_prediction"

    @property
    def pm(self) -> float:
        return 1.0 / self.n if self.mutation_rate is None else self.mutation_rate


@dataclass
class Dataset:
    features: np.ndarray          # runs x (T+1) x d
    best: np.ndarray              # runs x (T+1)
    diversity: np.ndarray         # runs x (T+1)
    target: np.ndarray            # runs x (T+1), current target occupancy


@dataclass
class Quantizer:
    continuous_edges: Tuple[np.ndarray, ...]
    continuous_indices: Tuple[int, ...]
    exact_indices: Tuple[int, ...]
    window_size: int

    def encode(self, row: np.ndarray, generation: int) -> Tuple[int, ...]:
        exact = tuple(int(round(row[i])) for i in self.exact_indices)
        quantized = tuple(
            int(np.searchsorted(edge, row[i], side="right"))
            for edge, i in zip(self.continuous_edges, self.continuous_indices)
        )
        window = () if self.window_size <= 0 else (generation // self.window_size,)
        return exact + quantized + window


@dataclass
class Model:
    states: List[Tuple[int, ...]]
    index: Dict[Tuple[int, ...], int]
    transition: csr_matrix
    best_readout: np.ndarray
    diversity_readout: np.ndarray
    target_mask: np.ndarray
    state_coordinates: np.ndarray


def bitcount(x: int) -> int:
    return int(x.bit_count())


def make_fitness(cfg: Config):
    n = cfg.n
    if cfg.problem == "onemax":
        return lambda x: float(bitcount(x)), float(n)
    if cfg.problem == "leadingones":
        def f(x: int) -> float:
            s = format(x, f"0{n}b")
            return float(len(s) - len(s.lstrip("1")))
        return f, float(n)
    if cfg.problem == "jump":
        k = cfg.jump_k
        def f(x: int) -> float:
            u = bitcount(x)
            return float(u if u == n or u <= n-k else n-u)
        return f, float(n)
    if cfg.problem == "trap":
        k = cfg.trap_k
        if n % k:
            raise ValueError("n must be divisible by trap_k")
        mask = (1 << k) - 1
        def f(x: int) -> float:
            total = 0
            for b in range(n // k):
                u = bitcount((x >> (b*k)) & mask)
                total += k if u == k else k - 1 - u
            return float(total)
        return f, float(n)
    raise ValueError(cfg.problem)


def tournament(pop: Sequence[int], values: np.ndarray, k: int, rng) -> int:
    pos = rng.integers(0, len(pop), size=k)
    sampled = values[pos]
    return int(pop[int(rng.choice(pos[sampled == sampled.max()]))])


def selected_parent_probs(values: np.ndarray, k: int) -> np.ndarray:
    mu = len(values)
    out = np.zeros(mu)
    # Exact for k=2 or 3; intended protocol uses k=2.
    for sample in np.ndindex(*(mu,) * k):
        sampled = values[list(sample)]
        winners = [j for j, p in enumerate(sample) if values[p] == sampled.max()]
        mass = (1 / mu) ** k / len(winners)
        for j in winners:
            out[sample[j]] += mass
    return out


def descriptors(pop: Sequence[int], fitness, cfg: Config) -> Tuple[np.ndarray, float, float]:
    values = np.asarray([fitness(x) for x in pop], dtype=float)
    best = float(values.max())
    best_count = int(np.sum(values == best))

    # Relative fitness histogram: counts at best, best-1, ..., and lower.
    relative = [int(np.sum(values == best-d)) for d in range(cfg.fitness_depth)]
    relative.append(int(np.sum(values < best-(cfg.fitness_depth-1))))

    # Locus-level allele frequencies and their histogram.
    p = np.zeros(cfg.n)
    for x in pop:
        for j in range(cfg.n):
            p[j] += (x >> j) & 1
    p /= cfg.mu
    allele_hist, _ = np.histogram(p, bins=np.linspace(0, 1, cfg.allele_hist_bins+1))

    counts = Counter(pop)
    pair_sum = 0.0
    genotypes = list(counts)
    for i, x in enumerate(genotypes):
        for y in genotypes[i+1:]:
            pair_sum += counts[x] * counts[y] * bitcount(x ^ y)
    diversity = pair_sum / (cfg.n * math.comb(cfg.mu, 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where((p == 0) | (p == 1), 0.0,
                       -p*np.log2(p) - (1-p)*np.log2(1-p))
    entropy = float(ent.mean())

    win = selected_parent_probs(values, cfg.tournament_k)
    selected_distance = 0.0
    for i, x in enumerate(pop):
        for j, y in enumerate(pop):
            selected_distance += win[i] * win[j] * bitcount(x ^ y) / cfg.n

    # Exact components first, continuous components last.
    row = np.asarray(
        [best, best_count, *relative, *allele_hist,
         values.mean(), values.var(), diversity, entropy, selected_distance],
        dtype=float,
    )
    return row, best, diversity


def simulate(cfg: Config, runs: int, seed: int) -> Dataset:
    fitness, optimum = make_fitness(cfg)
    rng = np.random.default_rng(seed)
    sample_row, _, _ = descriptors([0] * cfg.mu, fitness, cfg)
    features = np.empty((runs, cfg.generations+1, len(sample_row)), np.float32)
    best_arr = np.empty((runs, cfg.generations+1), np.float32)
    div_arr = np.empty_like(best_arr)
    target = np.empty_like(best_arr)

    for r in range(runs):
        pop = [int(rng.integers(0, 1 << cfg.n)) for _ in range(cfg.mu)]
        for t in range(cfg.generations+1):
            row, best, div = descriptors(pop, fitness, cfg)
            features[r, t], best_arr[r, t], div_arr[r, t] = row, best, div
            target[r, t] = float(best >= optimum - 1e-12)
            if t == cfg.generations:
                break
            vals = np.asarray([fitness(x) for x in pop])
            p1 = tournament(pop, vals, cfg.tournament_k, rng)
            p2 = tournament(pop, vals, cfg.tournament_k, rng)
            if rng.random() < cfg.crossover_rate:
                mask = int(rng.integers(0, 1 << cfg.n)); full = (1 << cfg.n)-1
                child = (p1 & mask) | (p2 & (~mask & full))
            else:
                child = p1 if rng.random() < 0.5 else p2
            for bit in range(cfg.n):
                if rng.random() < cfg.pm:
                    child ^= 1 << bit
            enlarged = pop + [child]
            evalues = np.asarray([fitness(x) for x in enlarged])
            worst = np.flatnonzero(evalues == evalues.min())
            del enlarged[int(rng.choice(worst))]
            pop = enlarged
    return Dataset(features, best_arr, div_arr, target)


def feature_layout(cfg: Config) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    # best, best_count, relative histogram, allele histogram are exact integers.
    exact_count = 2 + (cfg.fitness_depth + 1) + cfg.allele_hist_bins
    exact = tuple(range(exact_count))
    continuous = tuple(range(exact_count, exact_count + 5))
    return exact, continuous


def fit_quantizer(data: Dataset, cfg: Config, bins: int, window: int) -> Quantizer:
    exact, continuous = feature_layout(cfg)
    flat = data.features[:, :, continuous].reshape(-1, len(continuous))
    qs = np.linspace(0, 1, bins+1)[1:-1]
    edges = tuple(np.unique(np.quantile(flat[:, j], qs)) for j in range(flat.shape[1]))
    return Quantizer(edges, continuous, exact, window)


def encode_dataset(data: Dataset, q: Quantizer) -> List[List[Tuple[int, ...]]]:
    return [[q.encode(data.features[r, t], t) for t in range(data.features.shape[1])]
            for r in range(data.features.shape[0])]


def fit_model(train: Dataset, q: Quantizer, cfg: Config) -> Model:
    encoded = encode_dataset(train, q)
    states = sorted({z for traj in encoded for z in traj})
    index = {z:i for i,z in enumerate(states)}
    counts = defaultdict(Counter)
    best_sum = np.zeros(len(states)); div_sum = np.zeros(len(states)); visits = np.zeros(len(states))
    target_mask = np.zeros(len(states), dtype=bool)

    for r, traj in enumerate(encoded):
        for t, z in enumerate(traj):
            i = index[z]; visits[i] += 1
            best_sum[i] += train.best[r,t]; div_sum[i] += train.diversity[r,t]
            target_mask[i] |= bool(train.target[r,t])
            if t < cfg.generations:
                counts[i][index[traj[t+1]]] += 1

    best_readout = best_sum / visits
    div_readout = div_sum / visits
    rows=[]; cols=[]; vals=[]
    for i in range(len(states)):
        # Make aggregate target states absorbing for correct hitting probabilities.
        if target_mask[i]:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        row = counts[i]
        total = sum(row.values())
        if total == 0:
            rows.append(i); cols.append(i); vals.append(1.0); continue
        for j,c in row.items():
            if c >= cfg.min_transition_count:
                rows.append(i); cols.append(j); vals.append(c/total)
        # Preserve stochasticity if thresholding removed mass.
        used = sum(c for c in row.values() if c >= cfg.min_transition_count) / total
        if used < 1:
            rows.append(i); cols.append(i); vals.append(1-used)
    K = csr_matrix((vals,(rows,cols)), shape=(len(states),len(states)))
    coordinates = np.asarray(states, dtype=float)
    return Model(states,index,K,best_readout,div_readout,target_mask,coordinates)


def nearest_state(z: Tuple[int,...], model: Model) -> int:
    query = np.asarray(z,dtype=float)
    scale = np.maximum(1.0, np.ptp(model.state_coordinates, axis=0))
    return int(np.argmin(np.sum(np.abs(model.state_coordinates-query)/scale,axis=1)))


def initial_distribution(data: Dataset, q: Quantizer, model: Model) -> Tuple[np.ndarray,float]:
    d = np.zeros(len(model.states)); unseen=0
    for r in range(data.features.shape[0]):
        z=q.encode(data.features[r,0],0)
        i=model.index.get(z)
        if i is None:
            unseen += 1; i=nearest_state(z,model)
        d[i]+=1
    return d/d.sum(), unseen/data.features.shape[0]


def predict(initial: np.ndarray, model: Model, T: int) -> Dict[str,np.ndarray]:
    d=initial.copy(); out={k:np.empty(T+1) for k in ("best","diversity","success")}
    target=model.target_mask.astype(float)
    for t in range(T+1):
        out["best"][t]=d@model.best_readout
        out["diversity"][t]=d@model.diversity_readout
        out["success"][t]=d@target
        d=d@model.transition
    out["success"]=np.maximum.accumulate(out["success"])
    return out


def observed(data: Dataset) -> Dict[str,np.ndarray]:
    return {"best":data.best.mean(0), "diversity":data.diversity.mean(0),
            "success":np.maximum.accumulate(data.target.mean(0))}


def metrics(pred, obs):
    e=pred-obs
    return dict(rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(abs(e))),
                max_abs_error=float(np.max(abs(e))), final_error=float(e[-1]))


def validation_score(pred, obs, cfg):
    return (metrics(pred["best"],obs["best"])["rmse"]/cfg.n
            + metrics(pred["diversity"],obs["diversity"])["rmse"]
            + metrics(pred["success"],obs["success"])["rmse"])


def run(cfg: Config):
    print("Simulating independent train/validation/test splits...")
    train=simulate(cfg,cfg.train_runs,cfg.seed+1)
    valid=simulate(cfg,cfg.validation_runs,cfg.seed+2)
    test=simulate(cfg,cfg.test_runs,cfg.seed+3)
    candidates=[]
    obs_valid=observed(valid)
    for bins in cfg.candidate_bins:
        for window in cfg.candidate_windows:
            q=fit_quantizer(train,cfg,bins,window)
            m=fit_model(train,q,cfg)
            init,unseen=initial_distribution(valid,q,m)
            pred=predict(init,m,cfg.generations)
            score=validation_score(pred,obs_valid,cfg)
            candidates.append((score,bins,window,q,m,unseen))
            print(f"bins={bins} window={window:2d} states={len(m.states):6d} score={score:.6f} unseen={unseen:.3f}")
    score,bins,window,q,m,_=min(candidates,key=lambda x:x[0])
    init,unseen=initial_distribution(test,q,m)
    pred=predict(init,m,cfg.generations); obs=observed(test)

    prefix=Path(cfg.output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for t in range(cfg.generations+1):
        rows.append({"generation":t, **{f"predicted_{k}":pred[k][t] for k in pred},
                     **{f"observed_{k}":obs[k][t] for k in obs}})
    pd.DataFrame(rows).to_csv(prefix.with_name(prefix.name+"_trajectories").with_suffix(".csv"),index=False)
    metric_rows=[]
    for k in pred:
        metric_rows.append({"target":k,"selected_bins":bins,"selected_window":window,
                            "aggregate_states":len(m.states),"initial_unseen_rate":unseen,
                            **metrics(pred[k],obs[k])})
    pd.DataFrame(metric_rows).to_csv(prefix.with_name(prefix.name+"_metrics").with_suffix(".csv"),index=False)
    metadata={"config":{**asdict(cfg),"effective_mutation_rate":cfg.pm},
              "selected":{"bins":bins,"window":window,"validation_score":score,
                          "aggregate_states":len(m.states),"test_initial_unseen_rate":unseen}}
    prefix.with_name(prefix.name+"_metadata").with_suffix(".json").write_text(json.dumps(metadata,indent=2))

    x=np.arange(cfg.generations+1); fig,axes=plt.subplots(1,3,figsize=(15.5,4.6))
    for ax,k,ylabel in zip(axes,("best","diversity","success"),
                           ("Expected best fitness","Expected normalized diversity","Success probability")):
        ax.plot(x,obs[k],color="black",lw=2.4,label="Independent test simulation")
        ax.plot(x,pred[k],color="tab:red",lw=2,label="Rich aggregate model")
        ax.set_xlabel("Generation"); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle(f"{cfg.problem}: rich aggregate-state prediction, n={cfg.n}, mu={cfg.mu}, pc={cfg.crossover_rate}")
    fig.tight_layout(); fig.savefig(prefix.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("Selected bins/window:",bins,window,"states:",len(m.states))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem",choices=["onemax","leadingones","jump","trap"],default="onemax")
    p.add_argument("--n",type=int,default=20); p.add_argument("--mu",type=int,default=16)
    p.add_argument("--generations",type=int,default=100)
    p.add_argument("--train-runs",type=int,default=10000); p.add_argument("--validation-runs",type=int,default=2000)
    p.add_argument("--test-runs",type=int,default=10000); p.add_argument("--tournament-k",type=int,default=2)
    p.add_argument("--crossover-rate",type=float,default=.9); p.add_argument("--mutation-rate",type=float,default=None)
    p.add_argument("--jump-k",type=int,default=4); p.add_argument("--trap-k",type=int,default=4)
    p.add_argument("--seed",type=int,default=20260818); p.add_argument("--candidate-bins",type=int,nargs="+",default=[3,4,5])
    p.add_argument("--candidate-windows",type=int,nargs="+",default=[0,10,20])
    p.add_argument("--allele-hist-bins",type=int,default=5); p.add_argument("--fitness-depth",type=int,default=3)
    p.add_argument("--min-transition-count",type=int,default=1); p.add_argument("--output-prefix",default="rich_aggregate_ga_prediction")
    a=p.parse_args(); return Config(problem=a.problem,n=a.n,mu=a.mu,generations=a.generations,
        train_runs=a.train_runs,validation_runs=a.validation_runs,test_runs=a.test_runs,
        tournament_k=a.tournament_k,crossover_rate=a.crossover_rate,mutation_rate=a.mutation_rate,
        jump_k=a.jump_k,trap_k=a.trap_k,seed=a.seed,candidate_bins=tuple(a.candidate_bins),
        candidate_windows=tuple(a.candidate_windows),allele_hist_bins=a.allele_hist_bins,
        fitness_depth=a.fitness_depth,min_transition_count=a.min_transition_count,output_prefix=a.output_prefix)

if __name__=="__main__": run(parse_args())
