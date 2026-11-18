"""Vorgerechnete Messreihe (data/phn_results.json, erzeugt von tools/sweep.py): Zellen, Zugriff, gepoolte Kennzahlen.

Eine Zelle = eine Einstellung (ohne Zufalls-Seed); je Zelle zwei Mengen von je fünf Welten (A: Zufalls-Seed 1000, B: 2000). Kennzahlen sind
Mittel ± Streuung über die fünf Welten in Prozent, Einheit: Patienten (Recall@k) beziehungsweise Welten (Streuung)."""

import json
from functools import lru_cache
from pathlib import Path

import phn_constants as C
from phn_live import Settings

DATA = Path(__file__).resolve().parent / "data" / "phn_results.json"
BASE = Settings()


def sweep_settings() -> list:
    """Alle gemessenen Zellen: Voreinstellung und je eine Änderung (einfaktorielle Abweichungen); die Glättungen sind Ranker innerhalb der Zelle."""
    cells = [BASE]
    cells += [Settings(pr=1)]
    cells += [Settings(q=q) for q in (0.0, 0.6)]
    cells += [Settings(nd=nd) for nd in (50, 1000)]
    cells += [Settings(kb=kb) for kb in (1, 3, 10, 30)]
    cells += [Settings(a=a) for a in (0.5, 0.2)]
    cells += [Settings(fp=fp) for fp in (0.02, 0.05)]
    seen, out = set(), []
    for s in cells:
        if s.key() not in seen:
            seen.add(s.key())
            out.append(s)
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def cell(settings: Settings) -> dict | None:
    """Die gemessene Zelle zur Einstellung oder None, wenn sie nicht gemessen ist."""
    return load()["cells"].get(settings.key())


def stat(c: dict, which: str, ranker: str, k=5) -> tuple:
    """(Mittel, Streuung) einer Kennzahl aus Menge `which` ("A" oder "B") in Prozent."""
    m, sd = c[which]["summary"][ranker][str(k)]
    return m, sd


def pooled(c: dict, ranker: str, k=5) -> float:
    """Mittel der beiden Mengen (je fünf Welten): zehn Welten insgesamt."""
    return (stat(c, "A", ranker, k)[0] + stat(c, "B", ranker, k)[0]) / 2.0


def sets_agree(c: dict, ranker: str, k=5) -> float:
    """Unterschied der beiden Mengen in Punkten (Betrag), ein Maß für die Stabilität der Zahl."""
    return abs(stat(c, "A", ranker, k)[0] - stat(c, "B", ranker, k)[0])
