"""Kennzahlen und Verdikte (reine Funktionen: Art, Text), dazu der Beispielpatient. Jede Zahl im Text kommt aus der Messung."""

import numpy as np

import phn_constants as C
from phn_live import Measurement, Settings, best_simple, paired_diff, summarize  # noqa: F401  (Settings für die App)


def pct(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def spct(x: float) -> str:
    """Mit Vorzeichen (für Deltas: Streamlit färbt nach dem ersten Zeichen)."""
    return f"{x:+.1f}".replace(".", ",")


def dec(x: float, nd: int = 3) -> str:
    """Zahl mit Dezimalkomma; nur die Zahl wird umformatiert, nie ein ganzer Satz."""
    return f"{x:.{nd}f}".replace(".", ",")


def verdict_wissensbasis(kb: int, o5: float, nb5: float, simple_name: str, nb_minus_simple: float, se: float):
    """Verdikt zur Wissensbasis: Anteil von Naive Bayes am Orakel (Recall@5) und Vergleich mit der besten einfachen Regel (gepaart über die Welten)."""
    ratio = nb5 / o5 if o5 > 0 else float("nan")
    if ratio >= C.VERDICT_RATIO_OK:
        kind, lage = "success", "Die Wissensbasis reicht." if kb > 0 else "Die Wissensbasis ist bekannt."
    elif ratio >= C.VERDICT_RATIO_THIN:
        kind, lage = "info", "Die Wissensbasis ist dünn." if kb > 0 else "Die Wissensbasis ist bekannt; das Orakel kennt zusätzlich die Maske der geprüften Merkmale und die falschen Einträge."
    else:
        kind, lage = "warning", "Die Wissensbasis ist zu dünn." if kb > 0 else "Die Wissensbasis ist bekannt; das Orakel kennt zusätzlich die Maske der geprüften Merkmale und die falschen Einträge."
    text = f"Naive Bayes erreicht Recall@5 von {pct(nb5)} % und damit {100 * ratio:.0f} % des Orakels ({pct(o5)} %). {lage}"
    name = C.RANKER_NAMES[simple_name]
    if nb_minus_simple < -C.SD_FACTOR * se:
        text += f" Hier liegt die einfache Regel „{name}“ {pct(-nb_minus_simple)} Punkte vor Naive Bayes (mehr als zwei Standardfehler)."
    elif nb_minus_simple > C.SD_FACTOR * se:
        text += f" Naive Bayes liegt {pct(nb_minus_simple)} Punkte vor der besten einfachen Regel („{name}“, mehr als zwei Standardfehler)."
    else:
        text += f" Zwischen Naive Bayes und der besten einfachen Regel („{name}“) liegt kein Unterschied über zwei Standardfehlern ({pct(nb_minus_simple)} Punkte)."
    return kind, text


def verdict_abdeckung(a: float, kb: int, nb_minus_nbp: float, se: float):
    """Verdikt zu „nicht dokumentiert“: Naive Bayes (Abwesenheit zählt) gegen „nur dokumentierte Merkmale“, gepaarte Differenz Recall@5.

    Der Satz „nicht dokumentiert ist nicht abwesend“ steht nur, wenn nicht alle Merkmale geprüft wurden; bei voller Abdeckung liegt der Vorsprung der
    nur-dokumentierten-Regel an der geschätzten Wissensbasis (kb > 0), nicht an der Abdeckung."""
    if nb_minus_nbp < -C.SD_FACTOR * se:
        if a < 1.0:
            return "warning", (f"Nicht dokumentiert ist nicht abwesend: „nur dokumentierte Merkmale“ liegt {pct(-nb_minus_nbp)} Punkte vor Naive Bayes, "
                               "das Nicht-Dokumentiertes als abwesend liest (mehr als zwei Standardfehler).")
        faelle = "1 Fall" if kb == 1 else f"{kb} Fällen"
        herkunft = f"die Wissensbasis ist aus {faelle} je Krankheit geschätzt" if kb > 0 else "die Wissensbasis ist bekannt"
        return "warning", (f"„nur dokumentierte Merkmale“ liegt {pct(-nb_minus_nbp)} Punkte vor Naive Bayes, obwohl alle Merkmale geprüft wurden "
                           f"({herkunft}; mehr als zwei Standardfehler).")
    if nb_minus_nbp > C.SD_FACTOR * se:
        zusatz = " (alle Merkmale wurden geprüft)" if a >= 1.0 else ""
        return "success", f"Abwesenheit zählt als Evidenz{zusatz}: Naive Bayes liegt {pct(nb_minus_nbp)} Punkte vor „nur dokumentierte Merkmale“ (mehr als zwei Standardfehler)."
    return "info", f"Zwischen Naive Bayes und „nur dokumentierte Merkmale“ liegt kein Unterschied über zwei Standardfehlern ({pct(nb_minus_nbp)} Punkte)."


def verdict_seltene(auc_first: float, auc_last: float, ap_first: float, ap_last: float, prev_first: float, prev_last: float):
    """Verdikt zur seltenen Klasse: gleiche AUC, fallende Average Precision."""
    text = (f"Die AUC bleibt bei {dec(auc_last)} (bei Prävalenz {pct(100 * prev_first)} % waren es {dec(auc_first)}), die Average Precision fällt von "
            f"{dec(ap_first)} auf {dec(ap_last)} (Prävalenz {pct(100 * prev_last)} %).")
    return ("warning" if ap_last < 0.5 * ap_first else "info"), text


def kennzahlen(m: Measurement) -> dict:
    """Alle Kennzahlen der Hauptansicht aus einer Messung."""
    s = summarize(m)
    sname, sval = best_simple(s)
    d_simple = paired_diff(m, "NB", sname)
    d_abd = paired_diff(m, "NB", "NBp")
    return {"summary": s, "simple": sname, "simple_val": sval, "nb_minus_simple": d_simple, "nb_minus_nbp": d_abd}


def verdicts(settings: Settings, k: dict):
    s = k["summary"]
    v1 = verdict_wissensbasis(settings.kb, s["O"][5][0], s["NB"][5][0], k["simple"], *k["nb_minus_simple"])
    v2 = verdict_abdeckung(settings.a, settings.kb, *k["nb_minus_nbp"])
    return v1, v2


def example_view(m: Measurement, idx: int, top: int = 5) -> dict:
    """Beispielpatient idx (0-basiert, aus Welt 0): wahre Krankheit, dokumentierte und geprüfte Merkmale, die obersten `top` Krankheiten je Ranker."""
    ex = m.runs[0].example
    n = len(ex["d"])
    idx = int(min(max(idx, 0), n - 1))
    out = {"true": int(ex["d"][idx]), "features": [int(i) for i in np.flatnonzero(ex["x"][idx])], "checked": int(ex["mask"][idx].sum()), "top": {}}
    for r in C.RANKERS:
        sc = ex["scores"][r][idx]
        order = np.argsort(-sc, kind="stable")[:top]
        rank_true = int(1 + (sc > sc[out["true"]]).sum())
        out["top"][r] = {"ids": [int(i) for i in order], "scores": [float(sc[i]) for i in order], "rank_true": rank_true}
    from phn_rankers import posterior
    p = posterior(ex["scores"]["NB"][idx][None, :])[0]
    out["posterior_nb"] = {int(i): float(p[i]) for i in np.argsort(-p, kind="stable")[:top]}
    return out
