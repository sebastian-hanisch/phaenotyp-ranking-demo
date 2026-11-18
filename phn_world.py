"""Erzeuger der Welt: Krankheitsprofile, Hintergrund, Patienten, Wissensbasis aus wenigen Fällen (Annahmen: phn_constants, README „Modell“).

Jede Einheit ist eine Funktion mit ausdrücklichen Eingaben; der Zufall kommt nur über den Strom aus phn_stream (SplitMix64)."""

from dataclasses import dataclass

import numpy as np

import phn_constants as C
from phn_stream import Stream, derive


@dataclass(frozen=True, eq=False)
class World:
    """nd Krankheiten, ns Merkmale. `f` = Häufigkeit des Profilmerkmals bei Kranken (0 = nicht im Profil), `b` = Hintergrundrate je Merkmal,
    `g` = Wahrscheinlichkeit, dass das Merkmal bei einer Person mit der Krankheit dokumentiert wird (nach Dokumentationsverlust q), `prior` = Vorab."""

    nd: int
    ns: int
    q: float
    f: np.ndarray
    b: np.ndarray
    g: np.ndarray
    prior: np.ndarray


@dataclass(frozen=True, eq=False)
class Patients:
    d: np.ndarray   # wahre Krankheit je Patient
    x: np.ndarray   # dokumentierte Merkmale (0/1); nicht Geprüftes steht als 0
    mask: np.ndarray  # 1 = Merkmal wurde geprüft (Abdeckung); bei voller Abdeckung überall 1


def beliebtheit(st: Stream, ns: int) -> np.ndarray:
    """Beliebtheit der Merkmale ~ Zipf (Exponent POP_EXP), zufällig auf die Merkmale verteilt, Summe 1."""
    w = np.arange(1, ns + 1, dtype=float) ** (-C.POP_EXP)
    w = w / w.sum()
    return w[np.argsort(st.uniform(ns), kind="stable")]


def profile(st: Stream, nd: int, w: np.ndarray) -> np.ndarray:
    """Häufigkeit f[d, s] der Profilmerkmale: je Krankheit k_d Merkmale, gezogen ohne Zurücklegen mit Wahrscheinlichkeit proportional zur
    Beliebtheit (Efraimidis-Spirakis: die k größten Schlüssel log(u)/w), Häufigkeiten aus FREQ nach FREQ_P."""
    ns = len(w)
    keys = np.log(np.maximum(st.uniform((nd, ns)), 1e-300)) / w[None, :]
    order = np.argsort(-keys, axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")
    k = C.K_MIN_FEAT + np.minimum((st.uniform(nd) * (C.K_MAX_FEAT - C.K_MIN_FEAT + 1)).astype(int), C.K_MAX_FEAT - C.K_MIN_FEAT)
    chosen = rank < k[:, None]
    idx = np.minimum(np.searchsorted(np.cumsum(C.FREQ_P), st.uniform((nd, ns)), side="right"), len(C.FREQ) - 1)
    return np.where(chosen, np.asarray(C.FREQ)[idx], 0.0)


def hintergrund(w: np.ndarray) -> np.ndarray:
    return C.BG_BASE + C.BG_SLOPE * (w / w.max()) ** C.BG_POWER


def dokumentiert(f: np.ndarray, b: np.ndarray, q: float) -> np.ndarray:
    """g = 1 - (1 - f (1 - q)) (1 - b): vorhanden und aufgeschrieben (f (1 - q)) oder Hintergrund."""
    return 1.0 - (1.0 - f * (1.0 - q)) * (1.0 - b[None, :])


def vorab(st: Stream, nd: int, pr: int) -> np.ndarray:
    if pr == 0:
        return np.full(nd, 1.0 / nd)
    p = np.arange(1, nd + 1, dtype=float) ** (-C.PRIOR_ZIPF_EXP)
    p = p / p.sum()
    return p[np.argsort(st.uniform(nd), kind="stable")]


def make_world(seed: int, nd: int, q: float, pr: int) -> World:
    st = Stream(derive(seed, 1))
    w = beliebtheit(st, C.NS)
    f = profile(st, nd, w)
    b = hintergrund(w)
    return World(nd, C.NS, q, f, b, dokumentiert(f, b, q), vorab(st, nd, pr))


def patient_g(g: np.ndarray, fp: float) -> np.ndarray:
    """Dokumentationswahrscheinlichkeit der Patienten mit zusätzlichem falschen Eintrag (Rate fp je nicht vorhandenem Merkmal)."""
    return 1.0 - (1.0 - g) * (1.0 - fp)


def sample_patients(world: World, seed: int, n: int, a: float = 1.0, fp: float = 0.0) -> Patients:
    """n Patienten: Krankheit nach dem Vorab, Merkmale unabhängig mit patient_g; Abdeckung a: nur der Anteil a der Merkmale wird geprüft.
    Patienten ohne jedes dokumentierte Merkmal werden entfernt (es gäbe nichts zu ranken)."""
    st = Stream(derive(seed, 2))
    d = st.weighted_index(world.prior, n)
    x = (st.uniform((n, world.ns)) < patient_g(world.g, fp)[d]).astype(float)
    m = (st.uniform((n, world.ns)) < a).astype(float) if a < 1.0 else np.ones_like(x)
    x = x * m
    keep = x.sum(axis=1) > 0
    return Patients(d[keep], x[keep], m[keep])


def estimate_counts(world: World, seed: int, n_case: int) -> np.ndarray:
    """Wissensbasis aus n_case gezogenen Fällen je Krankheit: Zahl der Fälle, in denen das Merkmal dokumentiert war (nd x ns)."""
    st = Stream(derive(seed, 3))
    u = st.uniform((world.nd, n_case, world.ns))
    return (u < world.g[:, None, :]).sum(axis=1).astype(float)


def laplace_profile(counts: np.ndarray, n_case: int, a: float = C.LAPLACE_A) -> np.ndarray:
    """Geschätzte Häufigkeit mit Glättung a je Seite: (Zahl + a) / (n + 2a)."""
    return (counts + a) / (n_case + 2.0 * a)


def shrink_profile(counts: np.ndarray, n_case: int, a: float) -> np.ndarray:
    """Schrumpfung zum Mittel des Merkmals über alle Krankheiten, Pseudozählung a Fälle (Untergrenze 0,001 für das Mittel)."""
    m = np.maximum(counts.sum(axis=0) / (n_case * counts.shape[0]), 1e-3)
    return (counts + a * m[None, :]) / (n_case + a)


def knowledge_bases(world: World, seed: int, kb: int) -> dict:
    """Wissensbasen, die die Ranker benutzen: kb = 0 die wahren Häufigkeiten (für alle dieselben), sonst aus kb Fällen geschätzt, und zwar
    mit Laplace-Glättung ("lap") und mit Schrumpfung zum Merkmalsmittel mit den Pseudozählungen aus C.SHRINK_A ("s3", "s10"); alle aus denselben Fällen."""
    if kb == 0:
        return {"lap": world.g, "s3": world.g, "s10": world.g}
    counts = estimate_counts(world, seed, kb)
    return {"lap": laplace_profile(counts, kb), "s3": shrink_profile(counts, kb, float(C.SHRINK_A[0])), "s10": shrink_profile(counts, kb, float(C.SHRINK_A[1]))}
