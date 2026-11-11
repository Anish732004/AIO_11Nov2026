"""Reorder-point optimization: inventory simulation + Bayesian optimization (Optuna) vs grid sweep.

Policy (continuous review, checked once a day):
    if inventory position (on hand + on order) <= R_t:  order Q units (several Q if far below R_t)
    the order arrives after a random lead time; demand that cannot be served is lost.

Two policies, each with two decision variables:
    static   R_t = R                        variables (R, Q)
    dynamic  R_t = mu_t + k * sigma_t       variables (k, Q)
             mu_t, sigma_t = AI forecast of lead-time demand and its uncertainty (tab 2), so the
             reorder point moves with promotions and seasons; k plays the role of Z.

Objective: average daily cost = holding + lost-sales penalty + ordering, subject to a service
constraint (fill rate or cycle service level >= target).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class Costs:
    holding: float = 0.10     # per unit per day (~36 %/year on a 100-rupee item)
    stockout: float = 20.0    # per unit of lost sales (margin + goodwill)
    order: float = 1000.0     # per order placed (transport, handling, admin)


def simulate(demand: np.ndarray, reorder: np.ndarray, Q: float, lead: np.ndarray,
             costs: Costs, warmup: int = 14) -> dict:
    """Simulate K lead-time scenarios at once (rows of `lead`) over T days of actual demand.

    demand  : (T,) actual daily demand
    reorder : (T,) reorder point in force at the end of each day
    lead    : (K, T) lead time (days) of an order placed at the end of day t in scenario k
    An order placed at the end of day t with lead time l is received at the end of day t+l,
    so it must cover demand on days t+1 .. t+l, the same lead-time demand as in tab 2.
    """
    K, T = lead.shape
    rows = np.arange(K)
    pipeline = np.zeros((K, T + int(lead.max()) + 1))
    on_hand = np.full(K, reorder[0] + Q / 2)
    on_order = np.zeros(K)
    lost = np.zeros(K)
    hold = np.zeros(K)
    n_orders = np.zeros(K)
    stockout = np.zeros((K, T), dtype=bool)
    placed = []                                    # (k, t, l) of orders placed after warm-up
    for t in range(T):
        served = np.minimum(on_hand, demand[t])
        short = demand[t] - served
        on_hand -= served
        stockout[:, t] = short > 0
        arrived = pipeline[:, t]
        on_hand += arrived
        on_order -= arrived
        ip = on_hand + on_order
        n = np.where(ip <= reorder[t], np.floor((reorder[t] - ip) / Q) + 1, 0)
        pipeline[rows, t + lead[:, t]] += n * Q
        on_order += n * Q
        if t >= warmup:
            lost += short
            hold += on_hand
            n_orders += n > 0
            k_idx = np.nonzero(n > 0)[0]
            placed.extend(zip(k_idx, [t] * len(k_idx), lead[k_idx, t]))

    days = T - warmup
    total_demand = demand[warmup:].sum()
    # cycle service level: share of orders with no stockout while they were on the way
    cs = np.concatenate([np.zeros((K, 1)), np.cumsum(stockout, axis=1)], axis=1)
    ok = [cs[k, t + l + 1] - cs[k, t + 1] == 0 for k, t, l in placed if t + l < T]
    cost = (costs.holding * hold + costs.stockout * lost + costs.order * n_orders) / days
    return {
        "cost": float(cost.mean()),
        "holding": float((costs.holding * hold / days).mean()),
        "stockout": float((costs.stockout * lost / days).mean()),
        "ordering": float((costs.order * n_orders / days).mean()),
        "fill_rate": float(1 - lost.sum() / (total_demand * K)),
        "csl": float(np.mean(ok)) if ok else 1.0,
        "avg_on_hand": float((hold / days).mean()),
        "orders_per_month": float((n_orders / days).mean() * 30),
    }


def lead_scenarios(days, p, K: int, T: int, seed: int = 0) -> np.ndarray:
    """Common random numbers: every candidate policy faces the same lead-time draws."""
    return np.random.default_rng(seed).choice(days, size=(K, T), p=p).astype(int)


class Problem:
    """One optimization problem: a policy family, its bounds and the service constraint."""

    def __init__(self, policy: str, demand, mu, sigma, lead, costs: Costs, service: str, target: float):
        self.policy, self.demand, self.mu, self.sigma = policy, np.asarray(demand, float), mu, sigma
        self.lead, self.costs, self.service, self.target = lead, costs, service, target
        d = self.demand.mean()
        if policy == "static":
            self.names = ("R", "Q")
            self.bounds = ((float(mu.min()), float((mu + 5 * sigma).max())), (d, 30 * d))
        else:
            self.names = ("k", "Q")
            self.bounds = ((-1.0, 4.0), (d, 30 * d))

    def reorder_points(self, x1: float) -> np.ndarray:
        if self.policy == "static":
            return np.full(len(self.demand), x1)
        return self.mu + x1 * self.sigma

    def evaluate(self, x1: float, x2: float) -> dict:
        r = simulate(self.demand, self.reorder_points(x1), x2, self.lead, self.costs)
        r.update({self.names[0]: x1, self.names[1]: x2,
                  "service": r[self.service], "feasible": r[self.service] >= self.target})
        return r


def bayes_opt(problem: Problem, n_trials: int = 50, seed: int = 0) -> pd.DataFrame:
    """Optuna TPE (a Bayesian, model-based optimizer) with the service level as a constraint."""
    def objective(trial):
        x1 = trial.suggest_float(problem.names[0], *problem.bounds[0])
        x2 = trial.suggest_float(problem.names[1], *problem.bounds[1])
        r = problem.evaluate(x1, x2)
        trial.set_user_attr("r", r)
        trial.set_constraint("service", problem.target - r["service"])   # feasible when <= 0
        return r["cost"]

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return pd.DataFrame([t.user_attrs["r"] for t in study.trials])


def grid_search(problem: Problem, n: int = 20) -> pd.DataFrame:
    g1 = np.linspace(*problem.bounds[0], n)
    g2 = np.linspace(*problem.bounds[1], n)
    return pd.DataFrame([problem.evaluate(a, b) for a in g1 for b in g2])


def best_feasible(trials: pd.DataFrame):
    f = trials[trials["feasible"]]
    return None if f.empty else f.loc[f["cost"].idxmin()]


def best_so_far(trials: pd.DataFrame) -> np.ndarray:
    """Best feasible cost after each evaluation (NaN until the first feasible one)."""
    c = np.where(trials["feasible"], trials["cost"], np.inf)
    out = np.minimum.accumulate(c)
    return np.where(np.isinf(out), np.nan, out)
