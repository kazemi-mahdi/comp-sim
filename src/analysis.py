# Multi-rep analysis: run scenarios, compute 95% t-CIs, print theory vs. sim
from __future__ import annotations
import argparse
import math
from typing import Dict, Any, Tuple, List

import yaml
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt  # NEW: for plots

from .mm1_sim import run_one_rep, mm1_theory


# CI 
def mean_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    """
    95% two-sided t-interval for the mean of independent replications.
    Returns (mean, half_width).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    m = float(np.mean(x))
    if n == 1:
        return m, 0.0
    s = float(np.std(x, ddof=1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * s / math.sqrt(n)
    return m, half


# Run one scenario (all R replications)
def run_scenario(name: str,
                 lam: float,
                 mu: float,
                 N: int,
                 R: int,
                 tau: float | None,
                 base_seed: int) -> Dict[str, Any]:
    """
    Run all R replications for one (λ, μ, N, τ) scenario and summarize.

    What this function must do:
      1) For r = 0 up to R-1:
            choose a seed = base_seed + r
            call run_one_rep(...) once
            store the *mean* wait, *mean* sojourn, ρ̂, and (optionally) SLA fraction
      2) Use mean_ci(...) to compute (mean, half-width) for Wq, W, ρ̂, and SLA.
      3) Call mm1_theory(lam, mu) to get theoretical values.
      4) Compute relative errors for Wq and W.
      5) Return everything in a dict (see return at bottom).

    Hints:
       rep = run_one_rep(lambda_rate=lam, mu_rate=mu, N=N, seed=seed, tau=tau)
       mean wait of this replication: float(np.mean(rep.waits))
       mean sojourn of this replication: float(np.mean(rep.sojourns))
       ρ̂ of this replication: rep.rho_hat
       SLA fraction of this replication: rep.p_sla  (may be None)
    """

    # Basic sanity checks
    assert R > 0, "Number of replications R must be > 0."
    assert N > 0, "Number of jobs N must be > 0."

    # Per-replication logged metrics
    waits_means: List[float] = []  # mean Wq per replication
    soj_means: List[float] = []    # mean W per replication
    rho_hats:   List[float] = []   # utilization per replication
    sla_fracs:  List[float] = []   # SLA fraction per replication (if tau is not None)

    # ----------------------- TODO (loop over replications) -----------------------
    # For each replication r = 0..R-1:
    #   1) Choose a seed = base_seed + r
    #   2) Call run_one_rep(...)
    #   3) Append per-rep means into the lists above.

    # Example of what you want to end up with (pseudo-code):
    #   for r in range(R):
    #       seed = ...
    #       rep = ...
    #       waits_means.append(...)
    #       soj_means.append(...)
    #       rho_hats.append(...)
    #       if tau is not None:
    #           sla_fracs.append(...)

    # remove or comment the line
    raise NotImplementedError(
        "Implement the replication loop: choose seed, call run_one_rep, "
        "and fill waits_means / soj_means / rho_hats / sla_fracs."
    )
    
    ###########################################################################

    # After your loop, these should all have length R
    assert len(waits_means) == R, "You must push exactly one mean wait per replication."
    assert len(soj_means) == R, "You must push exactly one mean sojourn per replication."
    assert len(rho_hats) == R, "You must push exactly one rho_hat per replication."
    if tau is not None:
        assert len(sla_fracs) == R, "You must push exactly one SLA fraction per replication."

    # ----------------------- TODO (CIs using mean_ci) -----------------------
    # Use the helper mean_ci(...) to get (mean, half-width) for:
    #    waits_means -> (Wq_m, Wq_hw)
    #    soj_means   -> (W_m,  W_hw)
    #    rho_hats    -> (rho_m, rho_hw)
    #    sla_fracs   -> (sla_m, sla_hw) if tau is not None, else (None, None)


    # remove or comment the line below
    raise NotImplementedError(
        "Use mean_ci(...) on waits_means / soj_means / rho_hats / sla_fracs "
        "to compute (mean, half_width) for each metric."
    )
    ###########################################################################

    # ----------------------- TODO (theory & relative error) -----------------------
    # Get theory values from mm1_theory and compute relative errors:
    #   th = mm1_theory(lam, mu)
    #   rel_err_Wq = abs(Wq_m - th['Wq']) / th['Wq']   (if th['Wq'] > 0)
    #   rel_err_W  = abs(W_m  - th['W'])  / th['W']    (if th['W']  > 0)
    
    # remove or comment the line below
    raise NotImplementedError(
        "Call mm1_theory(lam, mu) and compute rel_err_Wq and rel_err_W "
        "using the simulated means and the theoretical values."
    )
    ###########################################################################

    # Return a dict with everything needed by print_table(...) and plot_results(...)
    return {
        "name": name,
        "lambda": lam,
        "mu": mu,
        "rho_theory": th["rho"],
        "Wq_theory": th["Wq"],
        "W_theory": th["W"],
        "Wq_hat": Wq_m, "Wq_hw": Wq_hw,
        "W_hat":  W_m,  "W_hw":  W_hw,
        "rho_hat": rho_m, "rho_hw": rho_hw,
        "rel_err_Wq": rel_err_Wq,
        "rel_err_W": rel_err_W,
        "SLA_tau": tau,
        "SLA_hat": sla_m, "SLA_hw": sla_hw,
        "N": N, "R": R,
    }


#  console table
def print_table(rows: List[Dict[str, Any]]) -> None:
    """
    Print a nicely aligned comparison table: theory vs. simulation.
    """
    header = (
        f"{'Scenario':<10}"
        f"{'λ':>7}{'μ':>7}{'ρ(th)':>9}"
        f"{'Wq(th)':>12}{'W(th)':>12}"
        f"{'Wq̂':>12}{'±CI':>9}"
        f"{'Ŵ':>12}{'±CI':>9}"
        f"{'ρ̂':>9}{'±CI':>9}"
        f"{'RelErr(Wq)':>13}{'RelErr(W)':>13}"
        f"{'SLA(τ)':>9}{'p̂(≤τ)':>11}{'±CI':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        def fmt_opt(x: Any) -> str:
            return "   —    " if x is None else f"{x:7.3f}"

        line = (
            f"{r['name']:<10}"
            f"{r['lambda']:7.3f}{r['mu']:7.3f}{r['rho_theory']:9.3f}"
            f"{r['Wq_theory']:12.4f}{r['W_theory']:12.4f}"
            f"{r['Wq_hat']:12.4f}{r['Wq_hw']:9.4f}"
            f"{r['W_hat']:12.4f}{r['W_hw']:9.4f}"
            f"{r['rho_hat']:9.4f}{r['rho_hw']:9.4f}"
            f"{r['rel_err_Wq']*100:13.2f}%{r['rel_err_W']*100:13.2f}%"
            f"{fmt_opt(r['SLA_tau'])}"
            f"{fmt_opt(r['SLA_hat'])}{fmt_opt(r['SLA_hw'])}"
        )
        print(line)


# Plotting helpers
def plot_results(rows: List[Dict[str, Any]], prefix: str = "mm1") -> None:
    """
    Create simple plots to visualize:
      - Wq (theory vs sim ± CI)
      - W  (theory vs sim ± CI)
      - ρ  (theory vs sim ± CI)
    and save them as PNG files.

    This makes it easier to *see* how load (ρ) changes performance.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))

    # Extract arrays
    Wq_th = np.array([r["Wq_theory"] for r in rows])
    W_th  = np.array([r["W_theory"] for r in rows])
    rho_th = np.array([r["rho_theory"] for r in rows])

    Wq_hat = np.array([r["Wq_hat"] for r in rows])
    W_hat  = np.array([r["W_hat"] for r in rows])
    rho_hat = np.array([r["rho_hat"] for r in rows])

    Wq_hw = np.array([r["Wq_hw"] for r in rows])
    W_hw  = np.array([r["W_hw"] for r in rows])
    rho_hw = np.array([r["rho_hw"] for r in rows])

    # --- Wq (mean wait in queue) ---
    plt.figure()
    plt.errorbar(x, Wq_hat, yerr=Wq_hw, fmt="o", label="Sim mean Wq ± CI")
    plt.plot(x, Wq_th, marker="s", linestyle="--", label="Theory Wq")
    plt.xticks(x, names)
    plt.xlabel("Scenario")
    plt.ylabel("Mean waiting time Wq")
    plt.title("Wait in Queue: Theory vs Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_Wq.png")

    # --- W (mean time in system) ---
    plt.figure()
    plt.errorbar(x, W_hat, yerr=W_hw, fmt="o", label="Sim mean W ± CI")
    plt.plot(x, W_th, marker="s", linestyle="--", label="Theory W")
    plt.xticks(x, names)
    plt.xlabel("Scenario")
    plt.ylabel("Mean time in system W")
    plt.title("Time in System: Theory vs Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_W.png")

    # --- ρ (utilization) ---
    plt.figure()
    plt.errorbar(x, rho_hat, yerr=rho_hw, fmt="o", label="Sim ρ̂ ± CI")
    plt.plot(x, rho_th, marker="s", linestyle="--", label="Theory ρ")
    plt.xticks(x, names)
    plt.xlabel("Scenario")
    plt.ylabel("Utilization ρ")
    plt.ylim(0, 1.05)
    plt.title("GPU Utilization: Theory vs Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_rho.png")

    




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG plots (only print the table).",
    )
    ap.add_argument(
        "--plot-prefix",
        default="mm1",
        help="Filename prefix for saved plots (default: 'mm1').",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg.get("base_seed", 12345))
    rows: List[Dict[str, Any]] = []
    for idx, sc in enumerate(cfg["scenarios"]):
        scenario_seed = base_seed + 10000 * idx
        # print(scenario_seed)
        row = run_scenario(
            name=sc["name"],
            lam=float(sc["lambda"]),
            mu=float(sc["mu"]),
            N=int(sc["N"]),
            R=int(sc["R"]),
            tau=(float(sc["tau"]) if "tau" in sc and sc["tau"] is not None else None),
            base_seed=scenario_seed,
        )
        rows.append(row)

    print_table(rows)

    if not args.no_plots:
        plot_results(rows, prefix=args.plot_prefix)


if __name__ == "__main__":
    main()
    