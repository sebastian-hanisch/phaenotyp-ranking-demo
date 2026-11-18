"""Erzeuger der Welt: Form, Wertebereiche, Wiederholbarkeit, Dokumentationsformel, Patienten (Abdeckung, falsche Einträge), Wissensbasis aus Fällen."""

import numpy as np
import pytest

import phn_constants as C
import phn_world as W


@pytest.fixture(scope="module")
def welt():
    return W.make_world(123, 200, 0.3, 0)


def test_form_und_wertebereiche(welt):
    assert welt.f.shape == welt.g.shape == (200, C.NS)
    assert set(np.unique(welt.f)) <= {0.0, *C.FREQ}
    assert welt.prior.sum() == pytest.approx(1.0)
    assert welt.g.min() >= C.BG_BASE and welt.g.max() < 1.0


def test_jede_krankheit_hat_vier_bis_zwoelf_profilmerkmale(welt):
    k = (welt.f > 0).sum(axis=1)
    assert k.min() >= C.K_MIN_FEAT and k.max() <= C.K_MAX_FEAT
    assert set(k) == set(range(C.K_MIN_FEAT, C.K_MAX_FEAT + 1))   # bei 200 Krankheiten kommen alle Werte vor (Zufalls-Seed 123)


def test_dokumentationsformel_von_hand():
    f = np.array([[0.6, 0.0]])
    b = np.array([0.1, 0.2])
    # 1 - (1 - 0,6 · 0,7) · (1 - 0,1) = 1 - 0,58 · 0,9 = 0,478; zweites Merkmal nur Hintergrund: 0,2
    assert W.dokumentiert(f, b, 0.3) == pytest.approx(np.array([[0.478, 0.2]]))


def test_hintergrund_wachsend_mit_der_beliebtheit():
    w = np.array([0.5, 0.25, 0.25])
    b = W.hintergrund(w)
    assert b[0] == pytest.approx(C.BG_BASE + C.BG_SLOPE)       # beliebtestes Merkmal: Verhältnis 1
    assert b[1] == pytest.approx(C.BG_BASE + C.BG_SLOPE * 0.25)  # halbe Beliebtheit: Quadrat 0,25


def test_gleicher_zufalls_seed_gleiche_welt_anderer_seed_andere_welt():
    a, b, c = W.make_world(5, 50, 0.3, 0), W.make_world(5, 50, 0.3, 0), W.make_world(6, 50, 0.3, 0)
    assert np.array_equal(a.g, b.g) and not np.array_equal(a.g, c.g)


def test_vorab_gleichverteilt_und_zipf():
    gleich = W.make_world(1, 40, 0.3, 0).prior
    zipf = W.make_world(1, 40, 0.3, 1).prior
    assert np.allclose(gleich, 1 / 40)
    assert zipf.sum() == pytest.approx(1.0)
    assert zipf.max() / zipf.min() == pytest.approx(40.0)   # Zipf-Exponent 1: größter durch kleinsten gleich der Zahl der Krankheiten


def test_patienten_ohne_dokumentiertes_merkmal_entfallen(welt):
    p = W.sample_patients(welt, 9, 3000, a=1.0, fp=0.0)
    assert (p.x.sum(axis=1) > 0).all()
    assert len(p.d) < 3000          # bei Dokumentationsverlust 0,3 bleiben Patienten ohne Eintrag übrig
    assert len(p.d) > 2800


def test_abdeckung_prueft_nur_einen_teil_der_merkmale(welt):
    p = W.sample_patients(welt, 9, 2000, a=0.5, fp=0.0)
    assert abs(p.mask.mean() - 0.5) < 0.02
    assert (p.x <= p.mask).all()    # nur Geprüftes kann dokumentiert sein
    p1 = W.sample_patients(welt, 9, 200, a=1.0, fp=0.0)
    assert (p1.mask == 1).all()


def test_falsche_eintraege_erhoehen_die_rate_nicht_vorhandener_merkmale():
    w = W.make_world(2, 100, 0.3, 0)
    p0 = W.sample_patients(w, 4, 3000, fp=0.0)
    p1 = W.sample_patients(w, 4, 3000, fp=0.2)
    assert p1.x.sum() > p0.x.sum() * 1.05
    assert W.patient_g(w.g, 0.2) == pytest.approx(1 - (1 - w.g) * 0.8)


def test_wissensbasis_aus_faellen_schaetzt_die_haeufigkeit(welt):
    counts = W.estimate_counts(welt, 3, 400)
    est = counts / 400
    assert np.abs(est - welt.g).mean() < 0.02
    assert counts.max() <= 400 and counts.min() >= 0


def test_wissensbasen_bekannt_sind_die_wahre_haeufigkeit_und_geschaetzt_sind_es_nicht(welt):
    bekannt = W.knowledge_bases(welt, 3, 0)
    assert bekannt["lap"] is welt.g and bekannt["s3"] is welt.g and bekannt["s10"] is welt.g
    k3 = W.knowledge_bases(welt, 3, 3)
    assert not np.array_equal(k3["lap"], welt.g)
    assert k3["lap"].min() >= 0.5 / 4 - 1e-12 and k3["lap"].max() <= 3.5 / 4 + 1e-12   # Laplace bei 3 Fällen: zwischen 0,125 und 0,875


def test_schrumpfung_und_laplace_unterscheiden_sich_und_nutzen_dieselben_faelle(welt):
    k = W.knowledge_bases(welt, 3, 3)
    assert not np.allclose(k["lap"], k["s3"]) and not np.allclose(k["s3"], k["s10"])
    counts = W.estimate_counts(welt, 3, 3)
    assert np.allclose(k["lap"], W.laplace_profile(counts, 3))
    assert np.allclose(k["s3"], W.shrink_profile(counts, 3, 3.0))
    assert np.allclose(k["s10"], W.shrink_profile(counts, 3, 10.0))
