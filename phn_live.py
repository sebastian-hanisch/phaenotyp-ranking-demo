"""Live-Messung: fünf Welten × Patienten für eine Einstellung, Auswertung über die Welten, Beispielpatienten.

`measure` ist deterministisch (Zufalls-Seed der Einstellung, SplitMix64) und hängt nicht von st.session_state ab."""

from dataclasses import dataclass, field

import numpy as np

import phn_constants as C
import phn_metrics as M
import phn_rankers as R
import phn_world as W
from phn_stream import derive


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Einstellung einer Messung. Nur mit Schlüsselwörtern erzeugen (ein eingefügtes Feld darf keine Werte still verschieben)."""

    nd: int = C.DEFAULT["nd"]
    kb: int = C.DEFAULT["kb"]
    q: float = C.DEFAULT["q"]
    a: float = C.DEFAULT["a"]
    fp: float = C.DEFAULT["fp"]
    pr: int = C.DEFAULT["pr"]
    seed: int = C.DEFAULT["seed"]

    def key(self) -> str:
        """Schlüssel der Zelle in data/phn_results.json (ohne den Zufalls-Seed)."""
        return f"nd{self.nd}_kb{self.kb}_q{self.q}_a{self.a}_fp{self.fp}_pr{self.pr}"


@dataclass(frozen=True, eq=False)
class WorldRun:
    metrics: dict          # Ranker -> {1, 5, 10, "mrr"} -> Feld über die Patienten
    kobs: np.ndarray       # Zahl dokumentierter Merkmale je Patient
    calib: dict            # "O" / "NB" / "NS3" / "NS10" -> (Spitzen-Wahrscheinlichkeit, Treffer 0/1)
    example: dict = field(default_factory=dict)   # erste Patienten: wahre Krankheit, Merkmale, Punktzahlen je Ranker (nur Welt 0)


@dataclass(frozen=True, eq=False)
class Measurement:
    settings: Settings
    n_patients: int
    runs: tuple


def run_world(s: Settings, i: int, n_patients: int, n_example: int = 0) -> WorldRun:
    wseed = derive(s.seed, 10 + i)
    world = W.make_world(wseed, s.nd, s.q, s.pr)
    pat = W.sample_patients(world, wseed, n_patients, a=s.a, fp=s.fp)
    kbs = W.knowledge_bases(world, wseed, s.kb)
    sc = R.all_scores(pat.x, pat.mask, W.patient_g(world.g, s.fp), kbs, world.prior)
    metrics = {r: M.metric_arrays(sc[r], pat.d) for r in C.RANKERS}
    calib = {}
    for r in ("O", "NB", "NS3", "NS10"):     # Kalibrierung: die App zeigt O und NB, die Schrumpfungen sind für den Vergleich der Richtung der Abweichung da
        p = R.posterior(sc[r])
        calib[r] = (p.max(axis=1), (sc[r].argmax(axis=1) == pat.d).astype(float))
    example = {}
    if n_example:
        example = {"d": pat.d[:n_example], "x": pat.x[:n_example], "mask": pat.mask[:n_example],
                   "scores": {r: sc[r][:n_example] for r in C.RANKERS}, "prior": world.prior, "f": world.f}
    return WorldRun(metrics, pat.x.sum(axis=1), calib, example)


def measure(s: Settings, n_worlds: int = C.N_WORLDS, n_patients: int = C.N_PATIENTS, n_example: int = 30) -> Measurement:
    runs = tuple(run_world(s, i, n_patients, n_example if i == 0 else 0) for i in range(n_worlds))
    return Measurement(s, n_patients, runs)


def summarize(m: Measurement) -> dict:
    """Ranker -> Kennzahl (1, 5, 10, "mrr") -> (Mittel, Streuung über die Welten) in Prozent."""
    out = {}
    for r in C.RANKERS:
        out[r] = {k: M.mean_sd([100.0 * run.metrics[r][k].mean() for run in m.runs]) for k in (1, 5, 10, "mrr")}
    return out


def paired_diff(m: Measurement, r1: str, r2: str, k: int = 5) -> tuple:
    """Gepaarte Differenz r1 minus r2 in Prozentpunkten Recall@k: Mittel über die Welten und Standardfehler (Streuung der Differenzen / Wurzel der Weltenzahl)."""
    d = [100.0 * (run.metrics[r1][k].mean() - run.metrics[r2][k].mean()) for run in m.runs]
    mean, sd = M.mean_sd(d)
    return mean, sd / float(len(d)) ** 0.5


def by_nobs(m: Measurement) -> dict:
    """Ranker -> Liste je Fach (Zahl dokumentierter Merkmale): (Anteil der Patienten in %, Recall@5 in %), Mittel über die Welten."""
    out = {r: [] for r in C.RANKERS}
    shares = []
    for lo, hi in C.BINS_NOBS:
        sel = [(run.kobs >= lo) & (run.kobs <= hi) for run in m.runs]
        shares.append(float(np.mean([s.mean() for s in sel])) * 100.0)
        for r in C.RANKERS:
            vals = [100.0 * run.metrics[r][5][s].mean() for run, s in zip(m.runs, sel) if s.any()]
            out[r].append(float(np.mean(vals)) if vals else float("nan"))
    return {"share": shares, "recall5": out}


def calibration(m: Measurement, ranker: str = "NB") -> dict:
    p = np.concatenate([run.calib[ranker][0] for run in m.runs])
    hit = np.concatenate([run.calib[ranker][1] for run in m.runs])
    return {"mean_p": float(p.mean() * 100.0), "hit": float(hit.mean() * 100.0), "bins": M.calibration_bins(p, hit)}


def share_few(m: Measurement, limit: int = 2) -> float:
    """Anteil der Patienten mit höchstens `limit` dokumentierten Merkmalen in % (Mittel über die Welten)."""
    return float(np.mean([(run.kobs <= limit).mean() for run in m.runs]) * 100.0)


def best_simple(summary: dict, k: int = 5) -> tuple:
    """Name und (Mittel, Streuung) der besseren der beiden einfachen Regeln nach Recall@k."""
    name = max(C.SIMPLE_RANKERS, key=lambda r: summary[r][k][0])
    return name, summary[name][k]
