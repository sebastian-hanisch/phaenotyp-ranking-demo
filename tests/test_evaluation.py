"""Verdikte (reine Funktionen, Schwellen und jede Zahl im Text an handgesetzten Werten), Beispielpatient, Zahlenformat."""

import pytest

import phn_constants as C
import phn_evaluation as ev
import phn_live as L


def test_zahlenformat_mit_komma_und_vorzeichen():
    assert ev.pct(12.34) == "12,3" and ev.spct(2.0) == "+2,0" and ev.spct(-0.04) == "-0,0" and ev.dec(0.95) == "0,950"


@pytest.mark.parametrize("nb,o,kind", [(76.0, 76.0, "success"), (69.0, 76.0, "success"), (68.0, 76.0, "info"), (38.0, 76.0, "info"), (37.0, 76.0, "warning")])
def test_verdikt_wissensbasis_schwellen_bei_90_und_50_prozent_des_orakels(nb, o, kind):
    # Grenzen: 0,9 · 76 = 68,4 und 0,5 · 76 = 38,0 (inklusive: genau 50 Prozent gilt noch als „dünn“)
    assert ev.verdict_wissensbasis(3, o, nb, "IC", 0.0, 1.0)[0] == kind


def test_verdikt_wissensbasis_grenzfaelle_exakt():
    assert ev.verdict_wissensbasis(3, 80.0, 72.0, "IC", 0.0, 1.0)[0] == "success"   # genau 90 Prozent: reicht (inklusive)
    assert ev.verdict_wissensbasis(3, 80.0, 40.0, "IC", 0.0, 1.0)[0] == "info"      # genau 50 Prozent: dünn (inklusive)
    assert ev.verdict_wissensbasis(3, 80.0, 39.9, "IC", 0.0, 1.0)[0] == "warning"


def test_verdikt_wissensbasis_bei_bekannter_wissensbasis_nennt_nicht_dass_sie_duenn_sei():
    kind, text = ev.verdict_wissensbasis(0, 80.0, 70.0, "IC", 5.0, 1.0)   # 87,5 Prozent des Orakels
    assert kind == "info" and "Wissensbasis ist bekannt" in text and "dünn" not in text and "Maske" in text
    kind, text = ev.verdict_wissensbasis(0, 80.0, 80.0, "IC", 12.0, 1.0)
    assert kind == "success" and "Wissensbasis ist bekannt." in text and "Maske" not in text


def test_verdikt_wissensbasis_nennt_zahlen_und_die_einfache_regel():
    kind, text = ev.verdict_wissensbasis(3, 76.5, 44.5, "IC", -4.9, 1.0)
    assert "44,5" in text and "76,5" in text and "58 %" in text and "Häufigkeit × Informationsgehalt" in text and "4,9 Punkte vor Naive Bayes" in text
    assert kind == "info"
    _, text2 = ev.verdict_wissensbasis(3, 76.5, 70.0, "IC", 8.0, 1.0)
    assert "8,0 Punkte vor der besten einfachen Regel" in text2
    _, text3 = ev.verdict_wissensbasis(3, 76.5, 70.0, "IC", 1.9, 1.0)
    assert "kein Unterschied über zwei Standardfehlern" in text3 and "1,9" in text3   # 1,9 < 2 · 1,0


def test_verdikt_wissensbasis_schwelle_zwei_standardfehler_ist_strikt():
    assert "kein Unterschied" in ev.verdict_wissensbasis(3, 80.0, 70.0, "OV", -2.0, 1.0)[1]     # genau zwei: nicht deutbar
    assert "vor Naive Bayes (mehr als" in ev.verdict_wissensbasis(3, 80.0, 70.0, "OV", -2.01, 1.0)[1]


def test_verdikt_abdeckung_faelle():
    kind, text = ev.verdict_abdeckung(0.2, 0, -4.4, 1.0)
    assert kind == "warning" and "4,4 Punkte vor Naive Bayes" in text and "nicht abwesend" in text
    kind, text = ev.verdict_abdeckung(1.0, 0, 7.3, 0.5)
    assert kind == "success" and "alle Merkmale wurden geprüft" in text and "7,3" in text
    kind, text = ev.verdict_abdeckung(0.5, 0, -0.2, 0.4)
    assert kind == "info" and "kein Unterschied" in text
    assert "alle Merkmale wurden geprüft" not in ev.verdict_abdeckung(0.5, 0, 7.0, 0.5)[1]


def test_verdikt_abdeckung_behauptet_bei_voller_abdeckung_nie_nicht_abwesend():
    kind, text = ev.verdict_abdeckung(1.0, 3, -4.5, 1.0)
    assert kind == "warning" and "nicht abwesend" not in text and "obwohl alle Merkmale geprüft wurden" in text and "aus 3 Fällen" in text
    assert "Wissensbasis ist bekannt" in ev.verdict_abdeckung(1.0, 0, -4.5, 1.0)[1]
    assert "aus 1 Fall je Krankheit" in ev.verdict_abdeckung(1.0, 1, -4.5, 1.0)[1]


def test_verdikt_seltene_klasse_formatiert_zahlen_einzeln():
    kind, text = ev.verdict_seltene(0.951, 0.950, 0.925, 0.014, 0.5, 0.001)
    assert kind == "warning"
    assert text == ("Die AUC bleibt bei 0,950 (bei Prävalenz 50,0 % waren es 0,951), die Average Precision fällt von 0,925 auf 0,014 (Prävalenz 0,1 %).")
    assert ev.verdict_seltene(0.95, 0.95, 0.9, 0.8, 0.5, 0.1)[0] == "info"   # fällt nur um ein Neuntel: keine Warnung


@pytest.fixture(scope="module")
def messung():
    return L.measure(L.Settings(seed=5), n_worlds=2, n_patients=300, n_example=8)


def test_beispielpatient_form_und_grenzen(messung):
    v = ev.example_view(messung, 0)
    assert set(v["top"]) == set(C.RANKERS) and len(v["top"]["O"]["ids"]) == 5
    assert v["features"] and all(0 <= f < C.NS for f in v["features"])
    assert v["checked"] == C.NS            # volle Abdeckung
    assert sum(v["posterior_nb"].values()) <= 1.0 + 1e-12
    assert ev.example_view(messung, 10_000)["true"] == ev.example_view(messung, 7)["true"]   # außerhalb: letzter Patient
    assert ev.example_view(messung, -5)["true"] == v["true"]


def test_rang_der_wahren_krankheit_passt_zur_rangliste(messung):
    for i in range(8):
        v = ev.example_view(messung, i)
        top_ids = v["top"]["O"]["ids"]
        if v["true"] in top_ids:
            assert v["top"]["O"]["rank_true"] <= 5
        else:
            assert v["top"]["O"]["rank_true"] > 5


@pytest.mark.parametrize("preset,v1,v2", [
    ("Wissensbasis bekannt", "success", "success"),
    ("Seltene Krankheit (3 Fälle)", "info", "warning"),
    ("Nur ein Fall", "warning", "warning"),
    ("Nicht alles geprüft", "info", "warning"),
    ("Viele Lücken", "success", "success"),
    ("Falsch dokumentiert", "success", "success"),
])
def test_jedes_preset_zeigt_die_erwartete_verdikt_zeile(preset, v1, v2):
    """DEMO-PLAYBOOK §11a Punkt 7: je Preset die erwartete Verdikt-Art an der Live-Messung mit dem Standard-Zufalls-Seed."""
    p = {k: v for k, v in C.PRESETS[preset].items()}
    s = L.Settings(**p)
    (k1, _), (k2, _) = ev.verdicts(s, ev.kennzahlen(L.measure(s)))
    assert (k1, k2) == (v1, v2)
