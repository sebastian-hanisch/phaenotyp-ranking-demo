"""Live-Messung: Wiederholbarkeit, Gleichheit Naive Bayes = Orakel, Zweig-Tests je Regler (jede Variante ändert mindestens einen Ranker), Auswertungsfunktionen."""

import numpy as np
import pytest

import phn_constants as C
import phn_live as L

KLEIN = dict(n_worlds=2, n_patients=400, n_example=5)   # kleine Messungen: schnell, nur für Zweig- und Formtests
BASIS = L.Settings(seed=11)


def _r5(s, **kw):
    sm = L.summarize(L.measure(s, **{**KLEIN, **kw}))
    return {r: sm[r][5][0] for r in C.RANKERS}


def test_gleicher_zufalls_seed_gleiche_messung_anderer_seed_andere():
    a, b = _r5(BASIS), _r5(BASIS)
    assert a == b
    assert a != _r5(L.Settings(seed=12))


def test_settings_nur_mit_schluesselwoertern():
    with pytest.raises(TypeError):
        L.Settings(200, 0, 0, 0.3, 1.0, 0.0, 0, 35)   # positional: nicht erlaubt (ein eingefügtes Feld würde Werte still verschieben)
    assert L.Settings(seed=3).key() == "nd200_kb0_q0.3_a1.0_fp0.0_pr0"


def test_naive_bayes_gleich_orakel_bei_bekannter_wissensbasis_voller_abdeckung_ohne_fehler():
    m = L.measure(BASIS, **KLEIN)
    for run in m.runs:
        for k in (1, 5, 10, "mrr"):
            assert np.array_equal(run.metrics["NB"][k], run.metrics["O"][k])


@pytest.mark.parametrize("aenderung", [dict(q=0.6), dict(q=0.0), dict(a=0.2), dict(fp=0.05), dict(pr=1), dict(kb=3), dict(kb=1), dict(nd=50)])
def test_jeder_regler_aendert_mindestens_einen_ranker(aenderung):
    """Zweig-Test (Nullspalten-Falle): jede Stufe weicht von der Voreinstellung auf mindestens einem Ranker ab."""
    base = _r5(BASIS)
    neu = _r5(L.Settings(seed=11, **aenderung))
    assert any(abs(neu[r] - base[r]) > 1e-9 for r in C.RANKERS), aenderung


def test_schrumpfung_weicht_bei_geschaetzter_wissensbasis_von_laplace_ab_und_gleicht_ihr_bei_bekannter():
    bekannt = _r5(BASIS)
    assert bekannt["NS3"] == bekannt["NB"] == bekannt["NS10"]
    geschaetzt = _r5(L.Settings(seed=11, kb=3))
    assert len({geschaetzt["NB"], geschaetzt["NS3"], geschaetzt["NS10"]}) == 3


def test_jede_regelart_weicht_auf_mindestens_einer_einstellung_von_jeder_anderen_ab():
    werte = {r: [] for r in C.RANKERS}
    for s in (L.Settings(seed=11), L.Settings(seed=11, kb=3), L.Settings(seed=11, a=0.2), L.Settings(seed=11, pr=1)):
        for r, v in _r5(s).items():
            werte[r].append(v)
    for i, r1 in enumerate(C.RANKERS):
        for r2 in C.RANKERS[i + 1:]:
            # Orakel und die Naive-Bayes-Varianten sind nur in der Voreinstellung gleich (eigener Test); über die vier Einstellungen weichen alle Paare ab
            assert werte[r1] != werte[r2], (r1, r2)


def test_orakel_und_naive_bayes_unterscheiden_sich_bei_abdeckung_und_falschen_eintraegen():
    for aenderung in (dict(a=0.2), dict(fp=0.05)):
        r = _r5(L.Settings(seed=11, **aenderung))
        assert r["O"] > r["NB"]    # das Orakel kennt Maske und Fehler


def test_ohne_vorab_gleich_mit_vorab_bei_gleichverteiltem_vorab_und_verschieden_bei_zipf():
    gleich = _r5(BASIS)
    assert gleich["NB"] == gleich["NB0"]
    zipf = _r5(L.Settings(seed=11, pr=1))
    assert zipf["NB"] > zipf["NB0"]


def test_summarize_form_und_nichtnegative_streuung():
    m = L.measure(BASIS, **KLEIN)
    s = L.summarize(m)
    assert set(s) == set(C.RANKERS)
    for r in C.RANKERS:
        assert set(s[r]) == {1, 5, 10, "mrr"}
        assert s[r][1][0] <= s[r][5][0] <= s[r][10][0] <= 100.0 and s[r][5][1] >= 0.0


def test_by_nobs_anteile_summieren_zu_100():
    m = L.measure(BASIS, **KLEIN)
    nob = L.by_nobs(m)
    assert sum(nob["share"]) == pytest.approx(100.0)
    assert len(nob["recall5"]["O"]) == len(C.BINS_NOBS)


def test_paired_diff_gegen_handrechnung():
    m = L.measure(BASIS, **KLEIN)
    diffs = [100 * (run.metrics["O"][5].mean() - run.metrics["IC"][5].mean()) for run in m.runs]
    mean, se = L.paired_diff(m, "O", "IC")
    assert mean == pytest.approx(np.mean(diffs))
    assert se == pytest.approx(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    assert L.paired_diff(m, "NB", "O")[0] == 0.0    # gleich bei bekannter Wissensbasis


def test_kalibrierung_des_orakels_stimmt_im_mittel():
    cal = L.calibration(L.measure(BASIS, n_worlds=3, n_patients=800, n_example=0), "O")
    assert abs(cal["mean_p"] - cal["hit"]) < 4.0     # genannte Wahrscheinlichkeit gegen Trefferquote, in Prozentpunkten


def test_share_few_wird_mit_dem_dokumentationsverlust_groesser():
    assert L.share_few(L.measure(L.Settings(seed=11, q=0.6), **KLEIN)) > L.share_few(L.measure(L.Settings(seed=11, q=0.0), **KLEIN))


def test_beispielpatienten_nur_in_der_ersten_welt():
    m = L.measure(BASIS, **KLEIN)
    assert len(m.runs[0].example["d"]) == 5 and m.runs[1].example == {}


def test_worst_case_am_regler_maximum_laeuft_durch():
    """Slider-Maximum (ND 1000, 30 Fälle, kleinste Abdeckung, falsche Einträge): Hänger oder Aufblähen fallen hier auf; keine Zeitassertion."""
    m = L.measure(L.Settings(seed=3, nd=1000, kb=30, a=0.2, fp=0.05, pr=1), n_worlds=1, n_patients=1500, n_example=3)
    assert np.isfinite(L.summarize(m)["NB"][5][0])
