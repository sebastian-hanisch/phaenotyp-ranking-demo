"""Kennzahlen: Recall@k mit Zufallsauflösung bei Gleichständen, reziproker mittlerer Rang, Kalibrierung, AUC, Average Precision, Wilson-Intervall."""

import numpy as np

import phn_constants as C


def rank_bounds(scores: np.ndarray, truth: np.ndarray):
    """Kleinster Rang (1 + Zahl strikt besserer Krankheiten) und Zahl der Gleichstände mit der wahren Krankheit (ohne sie selbst) je Patient."""
    n = len(truth)
    t = scores[np.arange(n), truth][:, None]
    return 1 + (scores > t).sum(axis=1), (scores == t).sum(axis=1) - 1


def recall_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Wahrscheinlichkeit je Patient, dass die wahre Krankheit unter den ersten k steht, wenn Gleichstände zufällig aufgelöst werden."""
    lo, ties = rank_bounds(scores, truth)
    return np.clip((k - lo + 1) / (ties + 1.0), 0.0, 1.0)


def reciprocal_rank(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1 / mittlerer Rang bei Gleichständen (Rang = kleinster Rang + Hälfte der Gleichstände). Das ist **nicht** der Erwartungswert von 1/Rang bei zufälliger Auflösung (der ist bei Gleichständen größer)."""
    lo, ties = rank_bounds(scores, truth)
    return 1.0 / (lo + ties / 2.0)


def metric_arrays(scores: np.ndarray, truth: np.ndarray) -> dict:
    """Recall@1, @5, @10 und reziproker mittlerer Rang je Patient (die Ränge werden nur einmal gerechnet)."""
    lo, ties = rank_bounds(scores, truth)
    out = {k: np.clip((k - lo + 1) / (ties + 1.0), 0.0, 1.0) for k in (1, 5, 10)}
    out["mrr"] = 1.0 / (lo + ties / 2.0)
    return out


def mean_sd(values) -> tuple:
    """Mittel und Standardabweichung (n - 1) über die Welten."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def calibration_bins(p_top: np.ndarray, hit: np.ndarray):
    """Je Fach der Spitzen-Wahrscheinlichkeit: Zahl der Patienten, mittlere genannte Wahrscheinlichkeit, tatsächliche Trefferquote."""
    out = []
    for lo, hi in zip(C.CALIB_EDGES[:-1], C.CALIB_EDGES[1:]):
        s = (p_top >= lo) & (p_top < hi)
        out.append((int(s.sum()), float(p_top[s].mean()) if s.any() else float("nan"), float(hit[s].mean()) if s.any() else float("nan")))
    return out


def auc_ap(score_pos: np.ndarray, score_neg: np.ndarray):
    """ROC-AUC über mittlere Ränge (Mann-Whitney, Gleichstände zählen halb) und Average Precision (Summe der Genauigkeiten an den Positiven / Zahl Positive)."""
    s = np.concatenate([score_pos, score_neg])
    y = np.concatenate([np.ones(len(score_pos)), np.zeros(len(score_neg))])
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    ap = float((tp / (tp + fp) * y_sorted).sum() / y_sorted.sum())
    sorted_asc = np.sort(s)
    lo = np.searchsorted(sorted_asc, s, side="left")
    hi = np.searchsorted(sorted_asc, s, side="right")
    ranks = (lo + hi + 1) / 2.0  # mittlerer Rang (1-basiert)
    n_pos, n_neg = len(score_pos), len(score_neg)
    auc = float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return auc, ap


def wilson(k, n, z: float = 1.96):
    """Wilson-Intervall für einen Anteil k / n (z = 1,96: 95 %); k darf ein Feld sein."""
    k = np.asarray(k, dtype=float)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return centre - half, centre + half
