"""Zufallsstrom: die vektorisierte Fassung liefert dieselben Zahlen wie die Skalarklasse SplitMix64 (phn_rng.py), und phn_rng.py ist die geprüfte Kopie der Vorlage."""

from pathlib import Path

import numpy as np
import pytest

from phn_rng import SplitMix64
from phn_stream import Stream, derive

VORLAGE = Path(__file__).resolve().parent.parent.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5, 2**64 - 1])
def test_vektorfassung_gleich_skalarklasse(seed):
    ref = SplitMix64(seed)
    expected = np.array([ref.random() for _ in range(300)])
    assert np.array_equal(Stream(seed).uniform(300), expected)   # bitgleich


def test_zwei_aufrufe_setzen_den_strom_fort():
    a, b = Stream(9), Stream(9)
    assert np.array_equal(np.concatenate([a.uniform(7), a.uniform(5)]), b.uniform(12))


def test_form_wird_zeilenweise_gefuellt():
    st = Stream(3)
    arr = st.uniform((4, 5))
    assert arr.shape == (4, 5)
    assert np.array_equal(arr.ravel(), Stream(3).uniform(20))


def test_gleichverteilt_im_halboffenen_intervall():
    u = Stream(11).uniform(100000)
    assert 0.0 <= u.min() and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005


def test_gewichtete_indizes_wie_skalar_choice():
    probs = np.array([0.2, 0.5, 0.3])
    ref = SplitMix64(4)
    expected = [ref.choice(list(probs)) for _ in range(200)]
    assert list(Stream(4).weighted_index(probs, 200)) == expected


def test_abgeleitete_zufalls_seeds_sind_getrennt_und_stabil():
    seeds = {derive(35, salt) for salt in range(1, 6)}
    assert len(seeds) == 5
    assert derive(35, 1) == derive(35, 1)
    assert derive(35, 1) != derive(36, 1)


@pytest.mark.skipif(not VORLAGE.exists(), reason="Vorlage nicht vorhanden (Repo einzeln geklont)")
def test_phn_rng_ist_die_unveraenderte_kopie_der_vorlage():
    assert (Path(__file__).resolve().parent.parent / "phn_rng.py").read_bytes() == VORLAGE.read_bytes()
