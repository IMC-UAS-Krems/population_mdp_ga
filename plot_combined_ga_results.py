#!/usr/bin/env python3
"""Combine GA population-MDP experiment CSV files into one figure and summary table.

Expected input columns (from ga_population_mdp_experiment.py):
    generation
    theory_expected_best
    mc_mean_best
    mc_se
    mc_ci95_lower
    mc_ci95_upper
    theory_expected_diversity  (optional for this plot)

Outputs:
    <output_prefix>.png          combined trajectory plot
    <output_prefix>.pdf          vector-format plot
    <output_prefix>_summary.csv  numerical summary table
    <output_prefix>_summary.tex  LaTeX table

Example:
    python plot_combined_ga_results.py \
        ga_mdp_onemax_mutation_only.csv ga_mdp_onemax.csv \
        --labels "Mutation only ($p_c=0$)" "Crossover ($p_c=0.9$)" \
        --output-prefix onemax_combined
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "generation",
    "theory_expected_best",
    "mc_mean_best",
    "mc_se",
    "mc_ci95_lower",
    "mc_ci95_upper",
}


@dataclass
class ExperimentSummary:
    configuration: str
    generations: int
    theoretical_initial_best: float
    theoretical_final_best: float
    monte_carlo_final_best: float
    final_mc_ci95_lower: float
    final_mc_ci95_upper: float
    rmse: float
    mae: float
    maximum_absolute_error: float
    coverage_95_percent: float


def latex_escape(text: str) -> str:
    """Escape ordinary text while preserving strings that appear to contain LaTeX."""
    if "$" in text or "\\" in text:
        return text
    replacements = {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def default_label(path: Path) -> str:
    label = path.stem
    label = re.sub(r"[_-]+", " ", label)
    return label.strip().title()


def read_experiment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )
    frame = frame.sort_values("generation").reset_index(drop=True)
    if frame["generation"].duplicated().any():
        raise ValueError(f"{path} contains duplicate generation values")
    numeric_columns = sorted(REQUIRED_COLUMNS)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def summarize(frame: pd.DataFrame, label: str) -> ExperimentSummary:
    residual = frame["mc_mean_best"] - frame["theory_expected_best"]
    inside = (
        (frame["theory_expected_best"] >= frame["mc_ci95_lower"])
        & (frame["theory_expected_best"] <= frame["mc_ci95_upper"])
    )
    final = frame.iloc[-1]
    return ExperimentSummary(
        configuration=label,
        generations=int(final["generation"]),
        theoretical_initial_best=float(frame.iloc[0]["theory_expected_best"]),
        theoretical_final_best=float(final["theory_expected_best"]),
        monte_carlo_final_best=float(final["mc_mean_best"]),
        final_mc_ci95_lower=float(final["mc_ci95_lower"]),
        final_mc_ci95_upper=float(final["mc_ci95_upper"]),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        mae=float(np.mean(np.abs(residual))),
        maximum_absolute_error=float(np.max(np.abs(residual))),
        coverage_95_percent=100.0 * float(np.mean(inside)),
    )


def summary_dataframe(summaries: Sequence[ExperimentSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Configuration": s.configuration,
                "T": s.generations,
                "Theory initial": s.theoretical_initial_best,
                "Theory final": s.theoretical_final_best,
                "MC final": s.monte_carlo_final_best,
                "Final MC 95% CI lower": s.final_mc_ci95_lower,
                "Final MC 95% CI upper": s.final_mc_ci95_upper,
                "RMSE": s.rmse,
                "MAE": s.mae,
                "Max. absolute error": s.maximum_absolute_error,
                "95% CI coverage (%)": s.coverage_95_percent,
            }
            for s in summaries
        ]
    )


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Summary of the comparison between exact population-MDP predictions and independently simulated genetic algorithm trajectories. Coverage is the percentage of generations for which the theoretical expected best fitness lies within the pointwise Monte Carlo $95\%$ confidence interval.}",
        r"    \label{tab:combined-ga-trajectory-summary}",
        r"    \begin{tabular}{lrrrrrr}",
        r"        \hline",
        r"        \textbf{Configuration} & \textbf{$T$} & \textbf{Theory final} & \textbf{MC final} & \textbf{RMSE} & \textbf{MAE} & \textbf{Coverage (\%)} \\",
        r"        \hline",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "        {} & {:d} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.1f} \\\\".format(
                latex_escape(str(row["Configuration"])),
                int(row["T"]),
                row["Theory final"],
                row["MC final"],
                row["RMSE"],
                row["MAE"],
                row["95% CI coverage (%)"],
            )
        )
    lines.extend([r"        \hline", r"    \end{tabular}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_combined(
    frames: Sequence[pd.DataFrame],
    labels: Sequence[str],
    output_prefix: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (frame, label) in enumerate(zip(frames, labels)):
        color = colors[index % len(colors)]
        generation = frame["generation"].to_numpy()
        theory = frame["theory_expected_best"].to_numpy()
        empirical = frame["mc_mean_best"].to_numpy()
        lower = frame["mc_ci95_lower"].to_numpy()
        upper = frame["mc_ci95_upper"].to_numpy()

        ax.plot(
            generation,
            theory,
            color=color,
            linewidth=2.2,
            label=f"{label}: population-MDP",
        )
        ax.plot(
            generation,
            empirical,
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label}: Monte Carlo",
        )
        ax.fill_between(
            generation,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
            label=f"{label}: Monte Carlo 95% CI",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Expected best fitness")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Experiment CSV files produced by ga_population_mdp_experiment.py",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Legend labels in the same order as the CSV files",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("combined_ga_results"),
        help="Output prefix without file extension",
    )
    parser.add_argument(
        "--title",
        default="Exact population-MDP predictions versus Monte Carlo simulation",
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.csv_files):
        raise ValueError("The number of --labels must equal the number of CSV files")

    labels: List[str] = (
        list(args.labels)
        if args.labels is not None
        else [default_label(path) for path in args.csv_files]
    )
    frames = [read_experiment(path) for path in args.csv_files]
    summaries = [summarize(frame, label) for frame, label in zip(frames, labels)]
    summary = summary_dataframe(summaries)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_combined(frames, labels, output_prefix, args.title)

    csv_output = output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv")
    tex_output = output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".tex")
    summary.to_csv(csv_output, index=False, float_format="%.8f", quoting=csv.QUOTE_MINIMAL)
    write_latex_table(summary, tex_output)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {csv_output}")
    print(f"Wrote {tex_output}")


if __name__ == "__main__":
    main()
