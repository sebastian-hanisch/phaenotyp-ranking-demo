"""Vektorisierter Zufallsstrom auf Basis von SplitMix64 (phn_rng.py): dieselben Zahlen wie die Skalarklasse, aber ganze Blöcke auf einmal.

Der SplitMix64-Strom ist zählerbasiert: die i-te Ausgabe hängt nur vom Anfangszustand und von i ab. Darum lässt er sich mit
uint64-Feldern berechnen (Ganzzahlarithmetik modulo 2^64, auf jeder Plattform bitgleich). tests/test_stream.py prüft die Gleichheit
mit der Skalarklasse. Kein NumPy-Zufall und kein hash() im Programm."""

import numpy as np

from phn_rng import MASK, SplitMix64

GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_TWO53 = float(1 << 53)


def derive(seed: int, salt: int) -> int:
    """Abgeleiteter Zufalls-Seed: getrennte Ströme für Welt, Patienten und Wissensbasis aus einem Zufalls-Seed."""
    return SplitMix64((seed * 1_000_003 + salt) & MASK).next_u64()


class Stream:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def _u64(self, n: int) -> np.ndarray:
        idx = np.arange(1, n + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + n * GAMMA) & MASK
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit, in der Form `shape` (Zahl oder Tupel)."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        n = int(np.prod(shape)) if shape else 1
        z = self._u64(n)
        return (z >> np.uint64(11)).astype(np.float64).reshape(shape) / _TWO53

    def weighted_index(self, probs: np.ndarray, size: int) -> np.ndarray:
        """Indizes nach den Wahrscheinlichkeiten `probs` (Summe 1), wie SplitMix64.choice: erster Index mit u < kumulierte Summe."""
        cum = np.cumsum(probs)
        idx = np.searchsorted(cum, self.uniform(size), side="right")
        return np.minimum(idx, len(probs) - 1)
