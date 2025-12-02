# SimPy M/M/1 (GPU job slot) — single-replication template

# goals:
#    See SimPy's core ideas: Environment, Resource(capacity=1), processes, timeouts
#    Record per-job timestamps to compute Wq (wait) and W (sojourn/sojourn time)
#    Estimate utilization ρ̂ = (total busy time) / (last departure time)

# Guardrails:
#     asserts check: positive rates/size, monotone timestamps, no negative waits and that departures == N

from __future__ import annotations
import simpy
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


# ------------------------------- Results container -------------------------------
@dataclass
class RepResult:
    """Outputs of one terminating replication."""
    waits: np.ndarray         # Wq_i = start_i - arrival_i   (per job)
    sojourns: np.ndarray      # W_i  = depart_i - arrival_i (per job)
    rho_hat: float            # utilization estimate in [0, 1]
    p_sla: Optional[float]    # fraction with Wq <= tau (None if tau is None)
    n_completed: int          # jobs completed (should equal N)
    horizon: float            # last departure time (simulated time)


# ------------------------------- Model definition -------------------------------
class MM1Simulation:
    """
    A minimal SimPy model for a single GPU slot (M/M/1).
    The "server" is the GPU with capacity=1, FIFO discipline.
    """
    def __init__(self, env: simpy.Environment, mu_rate: float):
        assert mu_rate > 0, "Service rate μ must be > 0."
        self.env = env
        self.server = simpy.Resource(env, capacity=1)  # one GPU slot
        self.mu = float(mu_rate)                       # jobs per time unit (minutes)

        # Logs for metrics (one value per job)
        self.arrivals: List[float] = []
        self.starts:   List[float] = []
        self.departs:  List[float] = []

        # For utilization: accumulate (service_start, service_end) for each job
        self._busy_spans: List[Tuple[float, float]] = []

    def customer(self, job_id: int, rng: np.random.Generator):
        """
        One job's life: arrive -> (maybe wait) -> start service -> finish.
        You MUST draw a service time from Exp(μ) and yield env.timeout(service).
        """
        # Arrival
        t_arr = self.env.now
        self.arrivals.append(t_arr)

        # Request the single-slot GPU
        with self.server.request() as req:
            yield req

            # Service starts now
            t_start = ...
            self.starts.append(t_start)

            # ----------------------- TODO (service draw) -----------------------
            # service time ~ Exp(μ). NumPy uses 'scale' = 1 / rate for Exponential.
            # hint: rng.exponential(scale=1.0 / self.mu)
            
            s = ...  # float service_time
            
            # then remove or comment the line below to test your function
            raise NotImplementedError("Draw service ~ Exp(mu): s = rng.exponential(1/self.mu)")
            #####################################################################

            # Advance simulated time by service duration
            yield self.env.timeout(float(s))

            # Departure
            t_end = ...
            self.departs.append(t_end)

            # For utilization estimate
            self._busy_spans.append((t_start, t_end))


# ------------------------------- One replication -------------------------------
def run_one_rep(lambda_rate: float,
                mu_rate: float,
                N: int,
                seed: int,
                tau: Optional[float] = None) -> RepResult:
    """
    Run one terminating replication of N jobs in an M/M/1 queue.

    Parameters
    ----------
    lambda_rate : float  (λ) arrivals per time unit, interarrivals ~ Exp(λ)
    mu_rate     : float  (μ) services per time unit, service time ~ Exp(μ)
    N           : int    number of *arrivals* to simulate (terminating)
    seed        : int    RNG seed (reproducibility)
    tau         : float | None   optional SLA threshold for waiting time

    Returns
    -------
    RepResult with raw per-job vectors (waits, sojourns) + utilization estimate.
    """
    # Basic input sanity checks (helpful error messages if misused)
    assert lambda_rate > 0, "Arrival rate λ must be > 0."
    assert mu_rate > 0, "Service rate μ must be > 0."
    assert N > 0, "N must be a positive integer."
    if tau is not None:
        assert tau >= 0, "SLA threshold τ must be nonnegative."

    # (Optional but recommended) stability check for educational runs
    # Comment out if you want to allow unstable experiments intentionally.
    assert lambda_rate < mu_rate, "For M/M/1 with finite means, require λ < μ."

    rng = np.random.default_rng(int(seed))
    env = ...
    model = MM1Simulation(env, mu_rate=mu_rate)

    # --------------------------- Arrivals process ---------------------------
    def arrivals(env: simpy.Environment):
        """
        Generate N arrivals with interarrivals ~ Exp(λ).
        For each arrival: wait for IAT, then start a customer process.
        """
        for i in range(N):
            # ----------------------- TODO (IAT draw) -----------------------
            # Interarrival time ~ Exp(λ). Replace ... with a correct draw.
            iat = ...  # float interarrival_time
            # Fill in missing code (...),
            # then remove or comment the line below to test your function
            raise NotImplementedError("Draw IAT ~ Exp(lambda): iat = rng.exponential(1/lambda_rate)")
            #################################################################

            # Advance time to next arrival, then create the job process
            yield env.timeout(float(iat))
            env.process(model.customer(job_id=i + 1, rng=rng)) # job 1, 2, 3, .... , N

    # Schedule the source and run until all processes complete
    env.process(arrivals(env))
    env.run()  # with a terminating source, this will finish

    # ------------------------------ Post-run checks ------------------------------
    n_completed = len(model.departs)
    assert n_completed == N, f"Completed {n_completed} != N={N} (did all jobs finish?)"

    # Convert logs to arrays for vectorized metrics
    arr = np.asarray(..., dtype=float)
    sts = np.asarray(...,   dtype=float)
    dps = np.asarray(...,  dtype=float)

    # Shape & length checks
    assert arr.shape == sts.shape == dps.shape, "arrival/start/depart arrays must align."
    assert np.all(np.isfinite(arr)) and np.all(np.isfinite(sts)) and np.all(np.isfinite(dps)), \
        "Timestamps contain non-finite values."

    # Monotonicity checks (per job)
    assert np.all(arr <= sts), "Negative waits found (arrival > start) — check your logic."
    assert np.all(sts <= dps), "Start after departure? — timestamps must be nondecreasing."

    # Non-decreasing across jobs (FIFO single server ⇒ departures are nondecreasing)
    assert np.all(np.diff(arr) >= 0) or True,  # arrivals are generated in time order
    assert np.all(np.diff(dps) >= 0), "Departures must be nondecreasing in time."

    # Per-job metrics
    waits    = ...           # Wq (waiting times)
    sojourns = ...           # W (cycle time)

    assert np.all(waits >= 0),    "Wait times must be >= 0."
    assert np.all(sojourns >= 0), "Sojourn times must be >= 0."

    # Horizon (last departure)
    horizon = float(dps[-1])
    assert horizon >= 0, "Horizon (last departure time) must be >= 0."

    # Utilization estimate: total busy time / horizon
    busy = 0.0
    for s, e in model._busy_spans:
        # Each (s,e) should satisfy 0 <= start <= end
        assert s <= e, "Busy span has start > end."
        busy += ...

    rho_hat = (busy / horizon) if horizon > 0 else 0.0
    # numeric guard to [0,1]
    rho_hat = float(max(0.0, min(1.0, rho_hat)))

    # SLA metric if tau provided
    p_sla: Optional[float] = None
    if tau is not None:
        p_sla = float(np.mean(waits <= float(tau))) if waits.size else 0.0
        assert 0.0 <= p_sla <= 1.0, "SLA fraction must be in [0,1]."

    return RepResult(
        waits=waits,
        sojourns=sojourns,
        rho_hat=rho_hat,
        p_sla=p_sla,
        n_completed=n_completed,
        horizon=horizon,
    )



def mm1_theory(lambda_rate: float, mu_rate: float) -> Dict[str, float]:
    """
    Classic steady-state M/M/1 formulas (require ρ = λ/μ < 1).

    Return a dict with keys: "rho", "Wq", "W", "Lq", "L".
    """
    lam = float(...)
    mu  = float(...)
    assert lam > 0 and mu > 0, "λ and μ must be positive."
    assert lam < mu, "M/M/1 finite means require λ < μ."

    # Traffic intensity ρ
    rho = lam / mu

    # ----------------------------- TODO(formulas) -----------------------------
    # Hints:
    #   Wq = λ / [ μ (μ - λ) ]
    #   W  = 1 / (μ - λ)
    #   Lq = ρ^2 / (1 - ρ)
    #   L  = ρ    / (1 - ρ)
    Wq = ...
    W  = ...
    Lq = ...
    L  = ...
    
    # then remove or comment the line below to test your function
    raise NotImplementedError("Fill M/M/1 formulas: Wq, W, Lq, L")
    ###########################################################################

    # Sanity checks: everything should be finite and positive
    for name, val in [("rho", rho), ("Wq", Wq), ("W", W), ("Lq", Lq), ("L", L)]:
        assert np.isfinite(val), f"{name} is not finite."
        assert val >= 0, f"{name} must be >= 0."

    return {"rho": rho, "Wq": Wq, "W": W, "Lq": Lq, "L": L}
