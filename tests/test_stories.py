"""Abnahmekriterien der Presets an der vorgerechneten Messreihe; jedes Kriterium kippt einzeln an einer künstlichen Zelle (Preset-Disziplin)."""

import phn_constants as C
import phn_results as RES
import phn_stories as ST


def test_alle_kriterien_aller_presets_gelten_in_beiden_mengen():
    for name, rows in ST.check_all().items():
        assert rows and {w for _, w, _ in rows} == {"A", "B"}, name
        assert all(ok for _, _, ok in rows), [(t, w) for t, w, ok in rows if not ok]


def test_jedes_preset_hat_kriterien():
    assert set(ST.CRITERIA) == set(C.PRESETS)


def _kuenstlich(o=76.0, nb=44.0, ic=49.0, ov=2.5, d_o_nb=0.0, d_nb_nbp=0.0, d_nb_ic=0.0, se=1.0):
    summary = {r: {"5": [v, 1.0]} for r, v in (("O", o), ("NB", nb), ("IC", ic), ("OV", ov), ("NBp", 0.0))}
    paired = {"O-NB": [d_o_nb, se], "NB-NBp": [d_nb_nbp, se], "NB-IC": [d_nb_ic, se]}
    return {"A": {"summary": summary, "paired": paired}, "B": {"summary": summary, "paired": paired}}


def test_bekannt_kippt_einzeln():
    ok = _kuenstlich(d_o_nb=0.0, d_nb_ic=12.0)
    assert all(v for _, _, v in ST.check_preset("Wissensbasis bekannt", ok))
    assert not ST.check_preset("Wissensbasis bekannt", _kuenstlich(d_o_nb=0.06, d_nb_ic=12.0))[0][2]
    assert not ST.check_preset("Wissensbasis bekannt", _kuenstlich(d_o_nb=0.0, d_nb_ic=9.99))[1][2]


def test_drei_faelle_kippt_einzeln():
    ok = _kuenstlich(o=76.0, nb=49.4, d_nb_ic=-3.0, se=1.0)                  # 49,4 = 65 Prozent von 76 (inklusive), −3 unter −2 · 1
    assert all(v for _, _, v in ST.check_preset("Seltene Krankheit (3 Fälle)", ok))
    assert not ST.check_preset("Seltene Krankheit (3 Fälle)", _kuenstlich(o=76.0, nb=49.5, d_nb_ic=-3.0))[0][2]
    assert not ST.check_preset("Seltene Krankheit (3 Fälle)", _kuenstlich(o=76.0, nb=44.0, d_nb_ic=-2.0, se=1.0))[1][2]   # genau zwei Standardfehler: nicht deutbar


def test_ein_fall_kippt_einzeln():
    ok = _kuenstlich(d_nb_ic=-8.0, ov=4.9)
    assert all(v for _, _, v in ST.check_preset("Nur ein Fall", ok))
    assert not ST.check_preset("Nur ein Fall", _kuenstlich(d_nb_ic=-7.9, ov=4.9))[0][2]
    assert not ST.check_preset("Nur ein Fall", _kuenstlich(d_nb_ic=-8.0, ov=5.0))[1][2]


def test_nicht_geprueft_kippt_einzeln():
    assert ST.check_preset("Nicht alles geprüft", _kuenstlich(d_nb_nbp=-2.1, se=1.0))[0][2]
    assert not ST.check_preset("Nicht alles geprüft", _kuenstlich(d_nb_nbp=-2.0, se=1.0))[0][2]


def test_viele_luecken_kippt_einzeln():
    assert ST.check_preset("Viele Lücken", _kuenstlich(o=55.0))[0][2]
    assert not ST.check_preset("Viele Lücken", _kuenstlich(o=55.1))[0][2]


def test_falsch_dokumentiert_kippt_einzeln():
    assert all(v for _, _, v in ST.check_preset("Falsch dokumentiert", _kuenstlich(d_o_nb=3.0, d_nb_nbp=5.0)))
    assert not ST.check_preset("Falsch dokumentiert", _kuenstlich(d_o_nb=3.1, d_nb_nbp=5.0))[0][2]
    assert not ST.check_preset("Falsch dokumentiert", _kuenstlich(d_o_nb=3.0, d_nb_nbp=4.9))[1][2]


def test_ein_kriterium_gilt_nur_wenn_es_in_beiden_mengen_gilt():
    c = {"A": _kuenstlich(o=50.0)["A"], "B": _kuenstlich(o=60.0)["B"]}   # getrennte Zellen: A erfüllt das Kriterium, B nicht
    rows = ST.check_preset("Viele Lücken", c)
    assert [ok for _, w, ok in rows] == [True, False]
