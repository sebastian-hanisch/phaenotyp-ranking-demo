"""Kennzahlen und Randfälle: Gleichstände, Rangsumme, Kalibrierung, leere Fächer, entartete Eingaben."""

import numpy as np
import pytest

import phn_metrics as M


def test_gleichstaende_werden_als_wahrscheinlichkeit_gezaehlt():
    # drei Krankheiten mit gleicher Punktzahl, die wahre ist eine davon: Recall@1 = 1/3, @2 = 2/3, @3 = 1
    sc = np.array([[1.0, 1.0, 1.0, 0.0]])
    t = np.array([2])
    assert M.recall_at_k(sc, t, 1)[0] == pytest.approx(1 / 3)
    assert M.recall_at_k(sc, t, 2)[0] == pytest.approx(2 / 3)
    assert M.recall_at_k(sc, t, 3)[0] == pytest.approx(1.0)
    assert M.reciprocal_rank(sc, t)[0] == pytest.approx(1 / 2)   # kleinster Rang 1, zwei Gleichstände: Rang 1 + 2/2 = 2


def test_alle_gleich_bewertet_ist_der_zufall():
    sc = np.zeros((5, 20))
    t = np.arange(5)
    assert np.allclose(M.recall_at_k(sc, t, 5), 5 / 20)


def test_recall_bei_k_gleich_zahl_der_krankheiten_ist_eins_und_waechst_mit_k():
    st = np.random.default_rng(0)  # nur Testdaten, kein Programm-Zufall
    sc = st.random((50, 12))
    t = st.integers(0, 12, 50)
    r = [M.recall_at_k(sc, t, k).mean() for k in (1, 5, 10, 12)]
    assert r == sorted(r) and r[-1] == 1.0


def test_metric_arrays_stimmt_mit_den_einzelfunktionen_ueberein():
    st = np.random.default_rng(1)
    sc = np.round(st.random((40, 15)), 1)   # Rundung erzeugt Gleichstände
    t = st.integers(0, 15, 40)
    ma = M.metric_arrays(sc, t)
    for k in (1, 5, 10):
        assert np.array_equal(ma[k], M.recall_at_k(sc, t, k))
    assert np.array_equal(ma["mrr"], M.reciprocal_rank(sc, t))


def test_rang_der_wahren_krankheit_von_hand():
    sc = np.array([[0.1, 0.9, 0.5, 0.3]])
    lo, ties = M.rank_bounds(sc, np.array([2]))
    assert (lo[0], ties[0]) == (2, 0)   # nur 0,9 ist besser


def test_mean_sd_mit_einem_wert_hat_streuung_null():
    assert M.mean_sd([3.0]) == (3.0, 0.0)
    assert M.mean_sd([1.0, 3.0]) == (2.0, pytest.approx(2 ** 0.5))


def test_kalibrierung_faecher_und_leere_faecher():
    p = np.array([0.1, 0.2, 0.6, 0.95, 0.999])
    hit = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    bins = M.calibration_bins(p, hit)
    assert [b[0] for b in bins] == [2, 1, 1, 1]   # [0;0,5) [0,5;0,9) [0,9;0,99) [0,99;1]
    assert bins[0][2] == pytest.approx(0.5)
    empty = M.calibration_bins(np.array([0.2]), np.array([1.0]))
    assert empty[1][0] == 0 and np.isnan(empty[1][1])


def test_wilson_enthaelt_den_anteil_und_bleibt_in_null_eins():
    lo, hi = M.wilson(np.array([0, 5, 10]), 10)
    assert (lo >= -1e-12).all() and (hi <= 1 + 1e-12).all() and (lo <= np.array([0, 0.5, 1])).all() and (hi >= np.array([0, 0.5, 1])).all()


def test_auc_bei_perfekter_und_umgekehrter_trennung():
    pos, neg = np.array([0.9, 0.8]), np.array([0.1, 0.2, 0.3])
    assert M.auc_ap(pos, neg) == (1.0, 1.0)
    assert M.auc_ap(neg, pos)[0] == 0.0
