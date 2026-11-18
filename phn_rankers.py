"""Ranker: Orakel (exakter Posterior des Erzeugers), Naive Bayes, nur dokumentierte Merkmale, Häufigkeit × Informationsgehalt, Zahl Übereinstimmungen.

Alle Funktionen liefern eine Punktzahl je Patient und Krankheit (Zeilen = Patienten); größer = wahrscheinlicher. Jede Einheit ist eine Regel in Worten."""

import numpy as np

import phn_constants as C


def _logs(g: np.ndarray):
    gc = np.clip(g, C.PROB_FLOOR, 1.0 - C.PROB_FLOOR)
    return np.log(gc), np.log(1.0 - gc)


def log_prior(prior: np.ndarray) -> np.ndarray:
    return np.log(prior)[None, :]


def informationsgehalt(ghat: np.ndarray) -> np.ndarray:
    """IC_s = -log(mittlere geschätzte Häufigkeit des Merkmals über alle Krankheiten), verschoben auf Minimum 0,001 (Phenomizer-artig)."""
    ic = -np.log(np.clip(ghat.mean(axis=0), C.PROB_FLOOR, None))
    return ic - ic.min() + 1e-3


def _nb_from_logs(x, lg, l1, lp):
    return x @ lg.T + (1.0 - x) @ l1.T + lp


def score_naive_bayes(x, ghat, prior):
    """Naive Bayes mit Vorab: log P(d) + Σ_s [x_s log g_ds + (1 - x_s) log (1 - g_ds)]; nicht Dokumentiertes zählt als abwesend."""
    lg, l1 = _logs(ghat)
    return _nb_from_logs(x, lg, l1, log_prior(prior))


def score_naive_bayes_ohne_vorab(x, ghat):
    lg, l1 = _logs(ghat)
    return _nb_from_logs(x, lg, l1, 0.0)


def score_nur_dokumentiert(x, ghat, prior):
    """Nur dokumentierte Merkmale zählen (Abwesenheit ist keine Evidenz): log P(d) + Σ_s x_s log g_ds."""
    lg, _ = _logs(ghat)
    return x @ lg.T + log_prior(prior)


def score_haeufigkeit_ic(x, ghat, ic):
    """Σ_s x_s · g_ds · IC_s: seltene Merkmale zählen mehr, Abwesenheit und Vorab zählen nicht."""
    return x @ (ghat * ic[None, :]).T


def score_uebereinstimmungen(x, ghat):
    """Zahl der dokumentierten Merkmale, die das Krankheitsprofil kennt (geschätzte Häufigkeit über SIMPLE_THRESHOLD)."""
    return x @ (ghat > C.SIMPLE_THRESHOLD).astype(float).T


def score_orakel(x, mask, g_patient, prior):
    """Exakter Posterior des Erzeugers: nur geprüfte Merkmale (mask = 1) tragen Evidenz, mit den wahren Häufigkeiten inklusive falscher Einträge."""
    lg, l1 = _logs(g_patient)
    return (mask * x) @ lg.T + (mask * (1.0 - x)) @ l1.T + log_prior(prior)


def posterior(scores: np.ndarray) -> np.ndarray:
    """Normierter Posterior je Patient aus Log-Punktzahlen (Zeilen summieren zu 1)."""
    z = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def all_scores(x, mask, g_patient, kbs, prior) -> dict:
    """Punktzahlen aller Ranker für dieselben Patienten (Schlüssel wie phn_constants.RANKERS); Logarithmen einmal gerechnet.
    `kbs` sind die Wissensbasen aus phn_world.knowledge_bases: Laplace ("lap") für Naive Bayes, die Variante ohne Vorab, nur dokumentierte, IC und Übereinstimmungen;
    Schrumpfung 3 und 10 nur für die beiden Naive-Bayes-Varianten NS3 und NS10."""
    ghat = kbs["lap"]
    lg, l1 = _logs(ghat)
    lp = log_prior(prior)
    ic = informationsgehalt(ghat)
    if g_patient is ghat or np.array_equal(g_patient, ghat):
        lgo, l1o = lg, l1
    else:
        lgo, l1o = _logs(g_patient)
    return {
        "O": (mask * x) @ lgo.T + (mask * (1.0 - x)) @ l1o.T + lp,
        "NB": _nb_from_logs(x, lg, l1, lp),
        "NS3": _nb_from_logs(x, *_logs(kbs["s3"]), lp),
        "NS10": _nb_from_logs(x, *_logs(kbs["s10"]), lp),
        "NB0": _nb_from_logs(x, lg, l1, 0.0),
        "NBp": x @ lg.T + lp,
        "IC": score_haeufigkeit_ic(x, ghat, ic),
        "OV": score_uebereinstimmungen(x, ghat),
    }
