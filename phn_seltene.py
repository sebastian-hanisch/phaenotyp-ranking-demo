"""Auswertung bei seltener Klasse: ROC-AUC gegen Average Precision über die Prävalenz, Wilson gegen Wald, Streuung der AUC bei wenig Positiven.

Scores ohne Normalverteilung (kein Logarithmus, kein Kosinus): Negative gleichverteilt in [0, 1), Positive U^(1/k). Dann gilt exakt
AUC = k / (k + 1); k = 19 gibt 0,95. Die Prävalenz ändert die Trennschärfe nicht, nur die Mischung der Klassen."""

import numpy as np

import phn_metrics as M
from phn_stream import Stream, derive

K_AUC = 19
TRUE_AUC = K_AUC / (K_AUC + 1.0)


def scores(st: Stream, n_pos: int, n_neg: int):
    neg = st.uniform(n_neg)
    pos = st.uniform(n_pos) ** (1.0 / K_AUC)
    return pos, neg


def prevalence_row(seed: int, prevalence: float, n_neg: int, reps: int) -> dict:
    """Mittel und Streuung über `reps` Ziehungen von AUC, Average Precision und PPV bei Sensitivität 90 %; n_pos aus der Prävalenz (mindestens 10)."""
    n_pos = max(10, int(round(n_neg * prevalence / (1.0 - prevalence))))
    st = Stream(derive(seed, 7))
    aucs, aps, ppvs = [], [], []
    for _ in range(reps):
        pos, neg = scores(st, n_pos, n_neg)
        auc, ap = M.auc_ap(pos, neg)
        thr = np.quantile(pos, 0.10)
        tp = float((pos >= thr).sum())
        fp = float((neg >= thr).sum())
        aucs.append(auc)
        aps.append(ap)
        ppvs.append(tp / (tp + fp))
    return {"prevalence": prevalence, "n_neg": n_neg, "n_pos": n_pos, "reps": reps,
            "auc": M.mean_sd(aucs), "ap": M.mean_sd(aps), "ppv": M.mean_sd(ppvs)}


def wilson_coverage(seed: int, n_pos: int, p_true: float = 0.9, reps: int = 20000) -> dict:
    """Überdeckung des 95-%-Intervalls für einen Anteil (wahr p_true): Wald gegen Wilson, mittlere Breite."""
    st = Stream(derive(seed, 8))
    k = (st.uniform((reps, n_pos)) < p_true).sum(axis=1).astype(float)
    p = k / n_pos
    half = 1.96 * np.sqrt(p * (1.0 - p) / n_pos)
    wald = float(((p - half <= p_true) & (p_true <= p + half)).mean())
    lo, hi = M.wilson(k, n_pos)
    wil = float(((lo <= p_true) & (p_true <= hi)).mean())
    return {"n_pos": n_pos, "reps": reps, "wald": wald, "wilson": wil, "width_wald": float((2 * half).mean()), "width_wilson": float((hi - lo).mean())}


def auc_spread(seed: int, n_pos: int, reps: int = 400, ratio: int = 99) -> dict:
    """Streuung der AUC-Schätzung eines festen Scores bei n_pos Positiven und ratio * n_pos Negativen (Prävalenz 1 %)."""
    st = Stream(derive(seed, 9))
    aucs = np.empty(reps)
    for i in range(reps):
        pos, neg = scores(st, n_pos, ratio * n_pos)
        aucs[i] = M.auc_ap(pos, neg)[0]
    return {"n_pos": n_pos, "reps": reps, "sd": float(aucs.std(ddof=1)), "mean": float(aucs.mean()),
            "share_off": float((np.abs(aucs - TRUE_AUC) > 0.03).mean() * 100.0)}
