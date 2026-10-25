"""Confidence calibration and its metrics.

A score of 0.9 should mean "right about 90 % of the time" across comparable cases.
Neural networks (and max-type scores like ``p_raw``) are usually over-confident, so
the raw score is mapped to a probability on a *separate calibration split*:

* **temperature scaling** - ``sigma(logit(p) / T)``; one parameter, keeps the ranking.
* **Platt scaling** - ``sigma(a * logit(p) + b)``; also corrects a shift (useful when
  negatives systematically get a non-trivial max-score).
* ``auto`` chooses between identity / temperature / Platt by k-fold cross-validated
  NLL *on the calibration split only* - the test split is never touched.

Metrics: Brier score, expected calibration error (equal-width and equal-mass bins),
NLL, AUROC, reliability-diagram tables and bootstrap confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-4


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def nll(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@dataclass
class Calibrator:
    method: str = "identity"
    T: float = 1.0
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0
    note: str = ""

    def __call__(self, p_raw):
        z = logit(p_raw)
        if self.method == "temperature":
            return sigmoid(z / self.T)
        if self.method == "platt":
            return sigmoid(self.a * z + self.b)
        return np.clip(np.asarray(p_raw, dtype=np.float64), 0.0, 1.0)

    def describe(self) -> str:
        if self.method == "temperature":
            return f"temperature scaling (T = {self.T:.3f})"
        if self.method == "platt":
            return f"Platt scaling (a = {self.a:.3f}, b = {self.b:.3f})"
        return "uncalibrated (identity)"

    def to_dict(self) -> dict:
        return {"method": self.method, "T": self.T, "a": self.a, "b": self.b, "n_fit": self.n_fit, "note": self.note,
                "description": self.describe()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(method=d.get("method", "identity"), T=float(d.get("T", 1.0)), a=float(d.get("a", 1.0)),
                   b=float(d.get("b", 0.0)), n_fit=int(d.get("n_fit", 0)), note=d.get("note", ""))


def fit_temperature(p_raw, y) -> float:
    z, y = logit(p_raw), np.asarray(y, dtype=np.float64)
    res = minimize_scalar(lambda lt: nll(sigmoid(z / np.exp(lt)), y), bounds=(-3.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def fit_platt(p_raw, y, l2: float = 1e-3) -> tuple[float, float]:
    z, y = logit(p_raw), np.asarray(y, dtype=np.float64)

    def obj(w):
        return nll(sigmoid(w[0] * z + w[1]), y) + l2 * (w[0] - 1.0) ** 2 + l2 * w[1] ** 2

    res = minimize(obj, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(1e-3, 50.0), (-50.0, 50.0)])
    return float(res.x[0]), float(res.x[1])


def _fit(method: str, p, y) -> Calibrator:
    if method == "temperature":
        return Calibrator("temperature", T=fit_temperature(p, y), n_fit=len(y))
    if method == "platt":
        a, b = fit_platt(p, y)
        return Calibrator("platt", a=a, b=b, n_fit=len(y))
    return Calibrator("identity", n_fit=len(y))


def fit_calibrator(p_raw, y, method: str = "auto", n_folds: int = 5, seed: int = 0) -> tuple[Calibrator, dict]:
    p_raw, y = np.asarray(p_raw, dtype=np.float64), np.asarray(y, dtype=np.int64)
    info: dict = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    if len(y) < 4 or y.min() == y.max():
        cal = Calibrator("identity", n_fit=len(y), note="calibration split has a single class - left uncalibrated")
        info["selected"] = "identity"
        info["warning"] = cal.note
        return cal, info
    if method != "auto":
        cal = _fit(method, p_raw, y)
        info["selected"] = method
        return cal, info
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, min(n_folds, len(y)))
    cv = {m: [] for m in ("identity", "temperature", "platt")}
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if y[tr].min() == y[tr].max():
            continue
        for m in cv:
            cal = _fit(m, p_raw[tr], y[tr])
            cv[m].append(nll(cal(p_raw[f]), y[f]) * len(f))
    scores = {m: float(np.sum(v) / len(y)) for m, v in cv.items() if v}
    best = min(scores, key=scores.get)
    info["cv_nll"] = {m: round(s, 4) for m, s in scores.items()}
    info["selected"] = best
    cal = _fit(best, p_raw, y)
    cal.note = f"selected by {len(folds)}-fold CV NLL on the calibration split"
    return cal, info


# --------------------------------------------------------------------------- metrics
def brier(p, y) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(((p - y) ** 2).mean())


def reliability_table(p, y, n_bins: int = 10, strategy: str = "uniform") -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        cnt.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else float("nan"))
        acc.append(float(y[m].mean()) if m.any() else float("nan"))
    return {"edges": edges.tolist(), "confidence": conf, "frequency": acc, "count": cnt}


def ece(p, y, n_bins: int = 10, strategy: str = "uniform") -> float:
    t = reliability_table(p, y, n_bins, strategy)
    n = max(1, sum(t["count"]))
    return float(sum(c * abs(a - f) for a, f, c in zip(t["confidence"], t["frequency"], t["count"]) if c > 0) / n)


def auroc(p, y) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def bootstrap_ci(fn, p, y, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    p, y = np.asarray(p), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(len(y), size=len(y))
        try:
            v = fn(p[i], y[i])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def calibration_report(p, y, n_bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "brier": brier(p, y), "ece": ece(p, y, n_bins),
           "ece_adaptive": ece(p, y, n_bins, "quantile"), "nll": nll(p, y), "auroc": auroc(p, y)}
    out["brier_ci"] = bootstrap_ci(brier, p, y, n_boot, seed)
    out["ece_ci"] = bootstrap_ci(lambda a, b: ece(a, b, n_bins), p, y, n_boot, seed)
    return out


def binary_entropy(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
