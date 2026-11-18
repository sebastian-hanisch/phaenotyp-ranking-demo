"""Seltene Klasse: Scores mit bekannter AUC, AP über die Prävalenz, Wilson-Überdeckung, Streuung der AUC."""

import pytest

import phn_seltene as S


def test_wahre_auc_ist_k_durch_k_plus_eins():
    assert S.TRUE_AUC == pytest.approx(0.95)
    r = S.prevalence_row(1, 0.5, 2000, 20)
    assert r["auc"][0] == pytest.approx(S.TRUE_AUC, abs=0.01)     # Mittel über 20 Ziehungen, Streuung je Ziehung etwa 0,004


def test_auc_bleibt_und_average_precision_faellt_mit_der_praevalenz():
    rows = [S.prevalence_row(2, p, 3000, 8) for p in (0.5, 0.1, 0.01)]
    assert max(abs(r["auc"][0] - S.TRUE_AUC) for r in rows) < 0.015
    assert rows[0]["ap"][0] > rows[1]["ap"][0] > rows[2]["ap"][0]


def test_zahl_der_positiven_aus_der_praevalenz_mindestens_zehn():
    assert S.prevalence_row(3, 0.001, 2000, 2)["n_pos"] == 10     # 2000 · 0,001 / 0,999 ≈ 2 → Untergrenze 10
    assert S.prevalence_row(3, 0.5, 400, 2)["n_pos"] == 400


def test_wilson_deckt_besser_als_wald_bei_wenigen_positiven():
    r = S.wilson_coverage(4, 10, reps=4000)
    assert r["wilson"] > r["wald"] + 0.15
    assert 0.9 < r["wilson"] <= 1.0
    r100 = S.wilson_coverage(4, 100, reps=4000)
    assert abs(r100["wilson"] - r100["wald"]) < 0.03


def test_streuung_der_auc_sinkt_mit_der_zahl_der_positiven():
    sds = [S.auc_spread(5, n, reps=150)["sd"] for n in (10, 40, 160)]
    assert sds[0] > sds[1] > sds[2]


def test_gleicher_zufalls_seed_gleiches_ergebnis():
    assert S.prevalence_row(6, 0.1, 1000, 3) == S.prevalence_row(6, 0.1, 1000, 3)
