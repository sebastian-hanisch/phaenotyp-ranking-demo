"""Gemeinsame Fixtures: die Mini-Welt, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mini-Welt: 3 Krankheiten, 4 Merkmale, 16 Muster. g[d][s] = Wahrscheinlichkeit, dass Merkmal s bei Krankheit d dokumentiert wird.
MINI_G = np.array([[0.90, 0.80, 0.10, 0.05],
                   [0.10, 0.60, 0.70, 0.10],
                   [0.05, 0.10, 0.20, 0.90]])
MINI_PRIOR = np.array([0.5, 0.3, 0.2])


@pytest.fixture
def mini_g():
    return MINI_G.copy()


@pytest.fixture
def mini_prior():
    return MINI_PRIOR.copy()


@pytest.fixture
def muster():
    """Alle 16 Muster dokumentierter Merkmale (Zeilen), geordnet nach der Binärzahl."""
    return np.array([[(i >> s) & 1 for s in range(4)] for i in range(16)], dtype=float)
