"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

- Posterior und Naive Bayes: schleifenweise Rechnung in reinem Python gegen die vektorisierten Ranker (Mini-Welt, alle 16 Muster).
- Recall: exakter Erwartungswert durch Aufzählung aller Muster gegen die Kennzahl-Funktion.
- AUC: paarweises Zählen; Average Precision: Definition an einer Handaufgabe; Wilson: Lösung der quadratischen Gleichung (andere Formel).
- Schätzung der Wissensbasis: Handrechnung."""

import numpy as np
import pytest

import phn_metrics as M
import phn_rankers as R
import phn_world as W
from phn_stream import Stream


def joint(g, prior, x, d):
    """P(d) * Π_s P(x_s | d) in reinem Python."""
    p = prior[d]
    for s in range(len(x)):
        p *= g[d][s] if x[s] == 1 else 1.0 - g[d][s]
    return p


def test_orakel_posterior_gleich_schleifenrechnung(mini_g, mini_prior, muster):
    post = R.posterior(R.score_orakel(muster, np.ones_like(muster), mini_g, mini_prior))
    for i, x in enumerate(muster):
        js = [joint(mini_g, mini_prior, x, d) for d in range(3)]
        for d in range(3):
            assert post[i, d] == pytest.approx(js[d] / sum(js), rel=1e-12)


def test_naive_bayes_mit_bekannter_wissensbasis_ist_das_orakel(mini_g, mini_prior, muster):
    a = R.score_naive_bayes(muster, mini_g, mini_prior)
    b = R.score_orakel(muster, np.ones_like(muster), mini_g, mini_prior)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_all_scores_stimmt_mit_den_einzelnen_rankern_ueberein(mini_g, mini_prior, muster):
    sc = R.all_scores(muster, np.ones_like(muster), mini_g, {"lap": mini_g, "s3": mini_g, "s10": mini_g}, mini_prior)
    assert np.allclose(sc["NS3"], sc["NB"]) and np.allclose(sc["NS10"], sc["NB"])   # bei bekannter Wissensbasis gleich
    assert np.allclose(sc["NB"], R.score_naive_bayes(muster, mini_g, mini_prior))
    assert np.allclose(sc["NB0"], R.score_naive_bayes_ohne_vorab(muster, mini_g))
    assert np.allclose(sc["NBp"], R.score_nur_dokumentiert(muster, mini_g, mini_prior))
    assert np.allclose(sc["IC"], R.score_haeufigkeit_ic(muster, mini_g, R.informationsgehalt(mini_g)))
    assert np.allclose(sc["OV"], R.score_uebereinstimmungen(muster, mini_g))


def test_erwarteter_recall_durch_aufzaehlung(mini_g, mini_prior, muster):
    """Erwartungswert über alle Muster und Krankheiten: Σ_x Σ_d P(x, d) · [d unter den ersten k]; die Reihenfolge ergibt die Schleife, ohne Gleichstände."""
    sc = R.score_orakel(muster, np.ones_like(muster), mini_g, mini_prior)
    for k in (1, 2, 3):
        exact = 0.0
        for i, x in enumerate(muster):
            js = [joint(mini_g, mini_prior, x, d) for d in range(3)]
            order = sorted(range(3), key=lambda d: -js[d])
            exact += sum(js[d] for d in order[:k])
        weights, hits = [], []
        for d in range(3):
            truth = np.full(16, d)
            hits.append(M.recall_at_k(sc, truth, k))
            weights.append(np.array([joint(mini_g, mini_prior, x, d) for x in muster]))
        measured = sum(float((w * h).sum()) for w, h in zip(weights, hits))
        assert measured == pytest.approx(exact, rel=1e-12)
        assert sum(float(w.sum()) for w in weights) == pytest.approx(1.0)


def test_orakel_mit_maske_ist_der_posterior_der_geprueften_merkmale(mini_g, mini_prior, muster):
    mask = np.array([1.0, 1.0, 0.0, 0.0])
    sc = R.score_orakel(muster * mask, np.tile(mask, (16, 1)), mini_g, mini_prior)
    post = R.posterior(sc)
    for i, x in enumerate(muster):
        js = [mini_prior[d] * (mini_g[d][0] if x[0] else 1 - mini_g[d][0]) * (mini_g[d][1] if x[1] else 1 - mini_g[d][1]) for d in range(3)]
        for d in range(3):
            assert post[i, d] == pytest.approx(js[d] / sum(js), rel=1e-12)


def test_auc_gegen_paarweises_zaehlen():
    st = Stream(7)
    for _ in range(25):
        pos = np.round(st.uniform(6), 1)   # Rundung erzeugt Gleichstände
        neg = np.round(st.uniform(9), 1)
        pair = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))
        assert M.auc_ap(pos, neg)[0] == pytest.approx(pair, rel=1e-12)


def test_average_precision_nach_definition_an_der_handaufgabe():
    """Rangliste 0,9 (+) 0,8 (−) 0,7 (+) 0,6 (−) 0,5 (+): Genauigkeit an den Positiven 1, 2/3, 3/5, Mittel (1 + 2/3 + 3/5) / 3."""
    auc, ap = M.auc_ap(np.array([0.9, 0.7, 0.5]), np.array([0.8, 0.6]))
    assert ap == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    assert auc == pytest.approx(3 / 6)  # Paare (+,−): 0,9 gewinnt beide (2), 0,7 gewinnt gegen 0,6 (1), 0,5 gewinnt keinen (0): 3 von 6


@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (9, 10), (10, 10), (27, 30), (50, 100)])
def test_wilson_gegen_loesung_der_quadratischen_gleichung(k, n):
    z = 1.96
    # Grenzen sind die Nullstellen von (1 + z²/n) p² − (2k/n + z²/n) p + (k/n)² = 0
    roots = np.sort(np.roots([1 + z * z / n, -(2 * k / n + z * z / n), (k / n) ** 2]))
    lo, hi = M.wilson(np.array([k]), n)
    assert lo[0] == pytest.approx(roots[0], abs=1e-12)
    assert hi[0] == pytest.approx(roots[1], abs=1e-12)


def test_schaetzung_der_wissensbasis_von_hand():
    counts = np.array([[0.0, 2.0, 3.0]])
    assert W.laplace_profile(counts, 3) == pytest.approx(np.array([[0.5 / 4, 2.5 / 4, 3.5 / 4]]))
    # Schrumpfung: Mittel der Zählungen je Merkmal über Krankheiten und Fälle, hier eine Krankheit: m = c / n, Untergrenze 0,001
    m = np.maximum(counts[0] / 3, 1e-3)
    assert W.shrink_profile(counts, 3, 2.0) == pytest.approx((counts + 2.0 * m) / (3 + 2.0))
