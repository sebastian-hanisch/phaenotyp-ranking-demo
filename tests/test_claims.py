"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a). Zahlen kommen aus data/phn_results.json (tools/sweep.py, Zufalls-Seed 1000 und 2000, je fünf Welten × 1500 Patienten);
ein Neu-Rechnen-Test (Band, nicht exakt: Linux rechnet an der letzten Binärstelle anders) prüft, dass die Datei zum Code passt.

Schreibweise: Recall@k in Prozent, Mittel der beiden Mengen A und B (zehn Welten) auf eine Stelle gerundet; „gepaart“ = Differenz der Rangmaße je Welt, Mittel ± Standardfehler."""

import pytest

import phn_constants as C
import phn_live as L
import phn_results as RES


def cell(**kw):
    return RES.cell(RES.Settings(**kw))


def pool(c, r, k=5):
    return round(RES.pooled(c, r, k), 1)


def pair(c, name):
    """((Mittel A, Mittel B), (Standardfehler A, Standardfehler B)) auf eine Stelle gerundet."""
    return tuple(round(c[w]["paired"][name][0], 1) for w in "AB"), tuple(round(c[w]["paired"][name][1], 1) for w in "AB")


def share(c, which="few"):
    return round((c["A"]["few"] + c["B"]["few"]) / 2, 1)


# --- Befund 1: Voreinstellung (Wissensbasis bekannt, 200 Krankheiten, q = 0,3, Vorab gleich, alle Merkmale geprüft) ---------------------------------

def test_befund_1_voreinstellung_recall():
    c = cell()
    assert {r: pool(c, r) for r in ("O", "NB", "NS3", "NS10", "NB0", "NBp", "IC", "OV")} == {"O": 76.5, "NB": 76.5, "NS3": 76.5, "NS10": 76.5, "NB0": 76.5, "NBp": 69.9, "IC": 64.3, "OV": 61.4}
    assert {r: pool(c, r, 1) for r in ("O", "NBp", "IC", "OV")} == {"O": 54.9, "NBp": 48.4, "IC": 41.3, "OV": 38.8}
    assert {r: pool(c, r, 10) for r in ("O", "NBp", "IC", "OV")} == {"O": 83.8, "NBp": 77.9, "IC": 74.2, "OV": 70.8}
    assert {r: pool(c, r, "mrr") for r in ("O", "NBp", "IC", "OV")} == {"O": 64.9, "NBp": 58.0, "IC": 51.8, "OV": 44.9}


def test_befund_1_naive_bayes_ist_in_jeder_welt_gleich_dem_orakel():
    c = cell()
    for w in "AB":
        assert c[w]["paired"]["O-NB"] == [0.0, 0.0]
        assert c[w]["summary"]["O"] == c[w]["summary"]["NB"]


def test_befund_1_gepaarte_differenzen():
    c = cell()
    assert pair(c, "NB-IC") == ((12.2, 12.1), (0.4, 0.6))
    assert pair(c, "NB-NBp") == ((6.4, 6.7), (0.3, 0.3))
    assert pair(c, "NB-OV") == ((15.4, 14.9), (0.4, 0.5))


# --- Befund 2: Vorab ------------------------------------------------------------------------------------------------------------------------------

def test_befund_2_vorab_nach_haeufigkeit():
    c = cell(pr=1)
    assert (pool(c, "O", 1), pool(c, "NB0", 1)) == (68.4, 57.1)
    assert (pool(c, "O"), pool(c, "NB0")) == (86.1, 77.6)
    assert pool(c, "NB", 1) == pool(c, "O", 1)
    assert (pool(cell(), "NB"), pool(cell(), "NB0")) == (76.5, 76.5)     # gleichverteiltes Vorab: Naive Bayes mit und ohne Vorab gleich


# --- Befund 3 und 4: Dokumentationsverlust, Zahl der Krankheiten ---------------------------------------------------------------------------------

def test_befund_3_dokumentationsverlust():
    assert [pool(cell(q=q), "O") for q in (0.0, 0.3, 0.6)] == [92.6, 76.5, 50.5]
    assert [share(cell(q=q)) for q in (0.0, 0.3, 0.6)] == [13.5, 25.5, 45.0]


def test_befund_4_zahl_der_krankheiten():
    assert [pool(cell(nd=n), "O") for n in (50, 200, 1000)] == [88.0, 76.5, 60.0]
    assert [pool(cell(nd=n), "O", 1) for n in (50, 200, 1000)] == [67.9, 54.9, 40.3]


# --- Befund 5: Zahl der dokumentierten Merkmale ----------------------------------------------------------------------------------------------------

def test_befund_5_recall_nach_zahl_der_merkmale():
    c = cell()
    mean = lambda key, sub=None: [round((a + b) / 2, 1) for a, b in zip((c["A"]["nobs"][key] if sub is None else c["A"]["nobs"][key][sub]), (c["B"]["nobs"][key] if sub is None else c["B"]["nobs"][key][sub]))]
    assert mean("share") == [25.5, 42.3, 29.0, 3.2]
    assert mean("recall5", "O") == [54.6, 77.8, 91.5, 97.8]
    assert mean("recall5", "NBp") == [43.5, 71.1, 88.4, 97.4]
    assert mean("recall5", "IC") == [40.1, 63.8, 82.9, 94.7]


# --- Befund 6: Wissensbasis aus wenigen Fällen ----------------------------------------------------------------------------------------------------

KB_ZAHLEN = {
    "NB": [21.4, 44.5, 65.3, 71.6], "NS3": [36.4, 54.9, 65.1, 67.5], "NS10": [35.3, 54.1, 66.5, 70.5],
    "NBp": [28.8, 49.0, 61.3, 65.7], "IC": [33.3, 49.4, 59.6, 62.7], "OV": [2.5, 2.5, 40.8, 51.4],
}


@pytest.mark.parametrize("ranker", list(KB_ZAHLEN))
def test_befund_6_recall_nach_faellen_je_krankheit(ranker):
    assert [pool(cell(kb=kb), ranker) for kb in (1, 3, 10, 30)] == KB_ZAHLEN[ranker]


def test_befund_6_orakel_bleibt_bei_76_5_unabhaengig_von_der_wissensbasis():
    assert {pool(cell(kb=kb), "O") for kb in (1, 3, 10, 30)} == {76.5}


def test_befund_6_naive_bayes_gegen_einfache_regel_kreuzt_zwischen_3_und_10_faellen():
    """gepaarte Differenz Naive Bayes (Laplace) minus Häufigkeit × Informationsgehalt, Mittel je Menge A / B; Standardfehler in Klammern der Prüfung."""
    d = {kb: pair(cell(kb=kb), "NB-IC") for kb in (1, 3, 10, 30)}
    assert d[1][0] == (-11.4, -12.6) and d[3][0] == (-4.6, -5.2) and d[10][0] == (5.9, 5.4) and d[30][0] == (9.0, 8.9)
    for kb in (1, 3):      # mehr als zwei Standardfehler vor Naive Bayes, in beiden Mengen
        assert all(m < -2 * se for m, se in zip(*d[kb]))
    for kb in (10, 30):    # Naive Bayes mehr als zwei Standardfehler vorn
        assert all(m > 2 * se for m, se in zip(*d[kb]))


def test_befund_6_mit_schrumpfung_liegt_naive_bayes_bei_1_und_3_faellen_vor_der_einfachen_regel():
    for kb, (a, b) in ((1, (3.0, 3.1)), (3, (5.8, 5.2))):
        (m, se) = pair(cell(kb=kb), "NS3-IC")
        assert m == (a, b) and all(x > 2 * s for x, s in zip(m, se))
    for kb, (a, b) in ((1, (1.9, 2.0)), (3, (4.8, 4.6))):
        (m, se) = pair(cell(kb=kb), "NS10-IC")
        assert m == (a, b) and all(x > 2 * s for x, s in zip(m, se))


def test_befund_6_schrumpfung_gegen_laplace():
    d3 = {kb: pair(cell(kb=kb), "NS3-NB") for kb in (1, 3, 10, 30)}
    assert d3[1][0] == (14.4, 15.6) and d3[3][0] == (10.4, 10.4) and d3[10][0] == (0.1, -0.4) and d3[30][0] == (-4.3, -4.0)
    assert all(m > 2 * se for m, se in zip(*d3[1])) and all(m > 2 * se for m, se in zip(*d3[3]))
    assert all(m < -2 * se for m, se in zip(*d3[30]))       # bei 30 Fällen ist Laplace besser als Schrumpfung 3
    assert pair(cell(kb=30), "NS10-NB")[0] == (-1.1, -1.1)    # und auch besser als Schrumpfung 10


# --- Befund 7: nicht jedes Merkmal wurde geprüft -----------------------------------------------------------------------------------------------------

def test_befund_7_abdeckung_recall():
    cs = [cell(a=a) for a in (1.0, 0.5, 0.2)]
    assert [pool(c, "O") for c in cs] == [76.5, 56.9, 41.8]
    assert [pool(c, "NB") for c in cs] == [76.5, 51.5, 34.3]
    assert [pool(c, "NBp") for c in cs] == [69.9, 51.7, 38.6]
    assert [share(c) for c in cs] == [25.5, 65.9, 92.9]


def test_befund_7_nicht_dokumentiert_ist_nicht_abwesend():
    half, fifth = cell(a=0.5), cell(a=0.2)
    assert pair(half, "NB-NBp") == ((-0.4, 0.1), (0.4, 0.5))            # halbe Abdeckung: kein Unterschied über zwei Standardfehlern
    assert all(abs(m) < 2 * se for m, se in zip(*pair(half, "NB-NBp")))
    assert pair(fifth, "NB-NBp") == ((-4.5, -4.2), (1.1, 0.5))          # ein Fünftel: „nur dokumentierte Merkmale“ liegt vorn
    assert all(m < -2 * se for m, se in zip(*pair(fifth, "NB-NBp")))
    assert pair(cell(), "NB-NBp")[0] == (6.4, 6.7)                       # bei voller Abdeckung liegt Naive Bayes vorn


# --- Befund 8: falsche Einträge ------------------------------------------------------------------------------------------------------------------------

def test_befund_8_falsche_eintraege():
    cs = [cell(fp=fp) for fp in (0.0, 0.02, 0.05)]
    assert [pool(c, "O") for c in cs] == [76.5, 67.4, 58.4]
    assert [pool(c, "NB") for c in cs] == [76.5, 66.8, 56.7]
    assert [pool(c, "NBp") for c in cs] == [69.9, 59.7, 49.8]
    assert pair(cs[1], "O-NB")[0] == (0.5, 0.7) and pair(cs[2], "O-NB")[0] == (1.8, 1.5)
    assert pair(cs[1], "NB-NBp")[0] == (7.4, 6.6) and pair(cs[2], "NB-NBp")[0] == (6.9, 7.0)


# --- Befund 9: Kalibrierung ----------------------------------------------------------------------------------------------------------------------------

def test_befund_9_kalibrierung_menge_a():
    base, kb3 = cell()["A"]["calib_NB"], cell(kb=3)["A"]["calib_NB"]
    assert (round(base["mean_p"], 1), round(base["hit"], 1)) == (55.5, 55.0)
    assert (round(kb3["mean_p"], 1), round(kb3["hit"], 1)) == (18.9, 25.4)
    n, genannt, treffer = kb3["bins"][1]       # Fach 0,5 bis unter 0,9
    assert (n, round(genannt, 3), round(treffer, 3)) == (415, 0.652, 0.875)
    for _, g, t in cell()["A"]["calib_NB"]["bins"]:
        assert abs(g - t) < 0.03               # bekannte Wissensbasis: in jedem der vier Fächer innerhalb 3 Punkten


# --- Stabilität der Messreihe ------------------------------------------------------------------------------------------------------------------------

def test_stabilitaet_zwei_mengen_von_welten():
    d = RES.load()
    worst = max(RES.sets_agree(c, r) for c in d["cells"].values() for r in C.RANKERS)
    assert round(worst, 2) == 3.02
    sds = [RES.stat(c, w, r, 5)[1] for c in d["cells"].values() for w in "AB" for r in C.RANKERS]
    assert round(max(sds), 1) == 5.7 and min(sds) < 1e-9     # kleinste Streuung: „Zahl Übereinstimmungen“ bei 1 oder 3 Fällen (alle Krankheiten gleich bewertet)
    assert d["meta"]["cells"] == 14 and d["meta"]["worlds_per_set"] == 5 and d["meta"]["patients_per_world"] == 1500


def test_neu_rechnen_voreinstellung_menge_a_im_band():
    """Datei gegen Neurechnung (Band 0,6 Punkte statt Gleichheit: BLAS und libm weichen unter Linux an der letzten Stelle ab)."""
    sm = L.summarize(L.measure(L.Settings(seed=C.SWEEP_SEEDS["A"]), n_example=0))
    c = cell()["A"]["summary"]
    for r in ("O", "NBp", "IC", "OV"):
        assert abs(sm[r][5][0] - c[r]["5"][0]) < 0.6, r


def test_neu_rechnen_drei_faelle_menge_a_im_band():
    sm = L.summarize(L.measure(L.Settings(kb=3, seed=C.SWEEP_SEEDS["A"]), n_example=0))
    c = cell(kb=3)["A"]["summary"]
    for r in ("NB", "NS3", "IC"):
        assert abs(sm[r][5][0] - c[r]["5"][0]) < 0.6, r


# --- Befund 10 bis 12: seltene Klasse ------------------------------------------------------------------------------------------------------------------

def test_befund_11_auc_und_average_precision():
    rows = RES.load()["seltene"]["prevalence"]
    assert [round(r["auc"][0], 3) for r in rows] == [0.951, 0.951, 0.950, 0.950]
    assert [round(r["ap"][0], 3) for r in rows] == [0.925, 0.590, 0.125, 0.014]
    assert [round(r["ppv"][0], 3) for r in rows] == [0.889, 0.472, 0.074, 0.008]
    assert [r["n_pos"] for r in rows] == [2000, 556, 202, 200] and [r["n_neg"] for r in rows] == [2000, 5000, 20000, 200000] and all(r["reps"] == 40 for r in rows)
    assert round(RES.load()["seltene"]["true_auc"], 2) == 0.95


def test_befund_12_wilson_gegen_wald():
    rows = RES.load()["seltene"]["wilson"]
    assert [round(r["wald"], 3) for r in rows] == [0.646, 0.810, 0.932]
    assert [round(r["wilson"], 3) for r in rows] == [0.930, 0.976, 0.936]
    assert [round(r["width_wald"], 3) for r in rows] == [0.279, 0.201, 0.116]
    assert [round(r["width_wilson"], 3) for r in rows] == [0.369, 0.215, 0.118]
    assert all(r["reps"] == 20000 for r in rows) and [r["n_pos"] for r in rows] == [10, 30, 100]


def test_befund_13_streuung_der_auc():
    rows = RES.load()["seltene"]["spread"]
    assert [round(r["sd"], 4) for r in rows] == [0.0156, 0.0095, 0.0048, 0.0028]
    assert [r["share_off"] for r in rows] == [5.5, 0.0, 0.0, 0.0]
    assert all(r["reps"] == 400 for r in rows) and [r["n_pos"] for r in rows] == [10, 30, 100, 300]


# --- Texte der App, die Zahlen der Konstanten nennen ----------------------------------------------------------------------------------------------------

def test_formelblock_der_app_nennt_die_konstanten_der_welt():
    from pathlib import Path
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert C.FREQ == (0.95, 0.6, 0.3, 0.1) and C.FREQ_P == (0.15, 0.25, 0.35, 0.25)
    assert "Häufigkeiten 0,95 / 0,6 / 0,3 / 0,1" in app and "mit den Anteilen 15 / 25 / 35 / 25 Prozent" in app
    assert (C.BG_BASE, C.BG_SLOPE, C.BG_POWER, C.POP_EXP, C.LAPLACE_A) == (0.01, 0.2, 2, 0.8, 0.5)
    assert r"b_s = 0{,}01 + 0{,}2\\,(w_s / w_{\\max})^2" in app and r"\\mathrm{Rang}^{-0{,}8}" in app and "c_{ds} + 0{,}5}{n + 1}" in app   # Quelltext hat doppelte Rückstriche
    assert C.SIMPLE_THRESHOLD == 0.05 and "Häufigkeit über 0,05" in app
    assert (C.K_MIN_FEAT, C.K_MAX_FEAT) == (4, 12) and "{C.K_MIN_FEAT} bis {C.K_MAX_FEAT}" in app    # die App liest die Zahlen aus den Konstanten
    assert (C.VERDICT_RATIO_OK, C.VERDICT_RATIO_THIN, C.SD_FACTOR) == (0.9, 0.5, 2.0)


# --- Nachträge der unabhängigen Aussagenprüfung: Zählungen, die im Text stehen -----------------------------------------------------------------------

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import phn_metrics as M  # noqa: E402
import phn_rankers as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
README_TEXT = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
APP_TEXT = (ROOT / "app.py").read_text(encoding="utf-8")
SEEDS = (C.SWEEP_SEEDS["A"], C.SWEEP_SEEDS["B"])


def measured(**kw):
    return [L.measure(L.Settings(seed=s, **kw), n_example=0) for s in SEEDS]


def ranked_share(**kw):
    """Anteil der gerankten Patienten (mindestens ein dokumentiertes Merkmal) in Prozent, Mittel über alle zehn Welten."""
    return float(np.mean([100.0 * len(run.kobs) / m.n_patients for m in measured(**kw) for run in m.runs]))


def calibration_pooled(ranker, **kw):
    """(mittlere genannte Wahrscheinlichkeit, Trefferquote) des Spitzenreiters in Prozent, über die zehn Welten gepoolt."""
    ms = measured(**kw)
    p = np.concatenate([run.calib[ranker][0] for m in ms for run in m.runs])
    h = np.concatenate([run.calib[ranker][1] for m in ms for run in m.runs])
    return 100.0 * float(p.mean()), 100.0 * float(h.mean())


def test_gerankte_patienten_anteile_stehen_in_readme_und_app():
    """Patienten ohne dokumentiertes Merkmal werden nicht gerankt; die Anteile stehen im README (Modell, Befunde 3 und 7) und in der App."""
    got = {"q0": ranked_share(q=0.0), "q3": ranked_share(), "q6": ranked_share(q=0.6), "a5": ranked_share(a=0.5), "a2": ranked_share(a=0.2)}
    assert {k: round(v, 1) for k, v in got.items()} == {"q0": 99.2, "q3": 97.8, "q6": 93.9, "a5": 84.9, "a2": 53.4}
    for text in ("99,2 / 97,8 / 93,9 Prozent", "97,8 / 84,9 / 53,4 Prozent"):
        assert text in README_TEXT
    for text in ("97,8 Prozent der Patienten in der Voreinstellung", "93,9 Prozent bei Dokumentationsverlust 0,6", "84,9 Prozent bei Abdeckung 0,5", "53,4 Prozent bei Abdeckung 0,2"):
        assert text in APP_TEXT


def test_zahl_uebereinstimmungen_2_5_ist_ein_artefakt_der_laplace_schaetzung():
    """Bei 1 und 3 Fällen ist die kleinste Laplace-Schätzung 0,5/(n+1) über der Schwelle 0,05: alle Krankheiten gleich bewertet, Recall@5 = 5/200; ab 10 Fällen darunter."""
    assert [round(C.LAPLACE_A / (n + 1), 3) for n in (1, 3, 10)] == [0.25, 0.125, 0.045]
    assert C.LAPLACE_A / 2 > C.SIMPLE_THRESHOLD and C.LAPLACE_A / 4 > C.SIMPLE_THRESHOLD and C.LAPLACE_A / 11 < C.SIMPLE_THRESHOLD
    assert 100.0 * 5 / 200 == 2.5
    for kb in (1, 3):
        c = cell(kb=kb)
        for w in "AB":
            mean, sd = RES.stat(c, w, "OV", 5)
            assert round(mean, 1) == 2.5 and sd < 1e-9
    assert pool(cell(kb=10), "OV") > 30.0       # ab 10 Fällen trennt die Regel
    for text in ("kleinste geschätzte Häufigkeit 0,25 beziehungsweise 0,125", "5 von 200 Krankheiten", "mit 0,045 unter der Schwelle"):
        assert text in README_TEXT


def test_kalibrierung_die_richtung_haengt_an_der_schaetzung_und_an_den_luecken():
    """Zehn Welten gepoolt (genannt, Treffer in Prozent), Band ±0,3 Punkte (BLAS/libm); die Zahlen stehen in Befund 9."""
    def near(got, want):
        return abs(got[0] - want[0]) < 0.3 and abs(got[1] - want[1]) < 0.3

    nb3, ns3, ns10 = (calibration_pooled(r, kb=3) for r in ("NB", "NS3", "NS10"))
    assert near(nb3, (18.6, 25.6)) and near(ns3, (35.3, 32.7)) and near(ns10, (16.0, 32.5))
    assert nb3[0] < nb3[1] and ns3[0] > ns3[1] and ns10[0] < ns10[1]       # beide Richtungen kommen vor
    a2, fp5, q6 = calibration_pooled("NB", a=0.2), calibration_pooled("NB", fp=0.05), calibration_pooled("NB", q=0.6)
    assert near(a2, (23.4, 14.3)) and near(fp5, (58.7, 33.4)) and near(q6, (28.4, 28.0))
    assert a2[0] > a2[1] + 5 and fp5[0] > fp5[1] + 10 and abs(q6[0] - q6[1]) < 1.0    # überkonfident bei Lücken in Abdeckung und bei falschen Einträgen, kalibriert bei reinem Verlust
    assert near(calibration_pooled("O", a=0.2), (17.6, 17.4)) and near(calibration_pooled("O", fp=0.05), (35.1, 34.8))
    for text in ("Add-½ 18,6 / 25,6 (unterkonfident)", "Schrumpfung 3 35,3 / 32,7 (überkonfident)", "Schrumpfung 10 16,0 / 32,5 (deutlich unterkonfident)", "23,4 bei 14,3", "58,7 bei 33,4", "28,4 bei 28,0", "17,6 bei 17,4", "35,1 bei 34,8"):
        assert text in README_TEXT


def test_kein_ranker_liegt_ueber_dem_orakel_ausser_um_rundungsreste():
    """896 Vergleiche (14 Zellen × 2 Mengen × 8 Ranker × 4 Kennzahlen): größter Überschuss über das Orakel 0,03 Punkte (Naive Bayes ohne Vorab, Rundung bei Gleichständen)."""
    d = RES.load()
    n, worst = 0, -1e9
    for c in d["cells"].values():
        for w in "AB":
            for k in ("1", "5", "10", "mrr"):
                o = c[w]["summary"]["O"][k][0]
                for r in C.RANKERS:
                    n += 1
                    worst = max(worst, c[w]["summary"][r][k][0] - o)
    assert n == 896 and round(worst, 2) == 0.03 and worst < 0.05
    assert "896 Vergleichen" in README_TEXT and "höchstens 0,03 Punkte" in README_TEXT


def test_informationsgehalt_wird_um_sein_minimum_verschoben_wie_beschrieben():
    ghat = np.array([[0.9, 0.1, 0.02], [0.5, 0.1, 0.02], [0.1, 0.3, 0.02]])
    ic = R.informationsgehalt(ghat)
    raw = -np.log(ghat.mean(axis=0))
    assert abs(ic.min() - 0.001) < 1e-12 and np.allclose(ic, raw - raw.min() + 0.001)
    assert "kleinste Wert 0,001" in README_TEXT and "+ 0{,}001" in APP_TEXT


def test_reziproker_mittlerer_rang_ist_nicht_der_erwartungswert_von_eins_durch_rang():
    scores = np.array([[2.0, 2.0, 2.0, 0.0]])       # wahre Krankheit 0, zwei Gleichstände auf Rang 1
    truth = np.array([0])
    assert M.reciprocal_rank(scores, truth)[0] == 0.5                     # 1/(1 + 2/2)
    erwartung = (1 + 1 / 2 + 1 / 3) / 3                                   # E[1/Rang] bei zufälliger Auflösung
    assert round(erwartung, 2) == 0.61 and M.reciprocal_rank(scores, truth)[0] < erwartung
    assert "reziproke mittlere Rang" in README_TEXT.replace("**", "")


def test_zwei_standardfehler_sind_bei_fuenf_welten_kein_95_prozent_kriterium():
    """Zweiseitiges p für t = 2 bei vier Freiheitsgraden (geschlossene Form der t-Verteilung, ohne SciPy): rund 0,12."""
    t = 2.0
    cdf = 0.5 + (3.0 / 8.0) * (t / math.sqrt(1 + t * t / 4)) * (1 - (t * t / (1 + t * t / 4)) / 12.0)
    p = 2 * (1 - cdf)
    assert round(p, 2) == 0.12
    assert "etwa 0,12" in README_TEXT


def test_streuung_der_auc_null_prozent_ist_beobachtet_nicht_nie():
    rows = RES.load()["seltene"]["spread"]
    erwartet = [400 * math.erfc(0.03 / (r["sd"] * math.sqrt(2))) for r in rows]      # Normalnäherung der AUC-Schätzung
    assert round(erwartet[1], 1) == 0.6 and erwartet[2] < 0.01 and erwartet[3] < 0.01
    assert "etwa 0,6 von 400" in README_TEXT


def test_readme_nennt_keine_planungsordner_oder_planbegriffe():
    for text in ("seltene-krankheiten-planung", "plan_phaenotyp", "Plan AP", "Der Plan sah", "des Plans", "gegenüber dem Plan", "Vormessung"):
        assert text not in README_TEXT, text
