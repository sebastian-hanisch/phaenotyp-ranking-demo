"""Abnahmekriterien der Presets: jedes Preset muss an der vorgerechneten Messreihe zeigen, wofür es steht (DEMO-PLAYBOOK §7, Preset-Disziplin).

Ein Kriterium ist eine Zeile (Text, Aussage über eine Zelle und eine Menge von Welten). Es gilt nur, wenn es in BEIDEN Mengen (A und B) gilt; Vorzeichen-
Kriterien über Rangunterschiede tragen die Bedingung „mehr als zwei Standardfehler“ der gepaarten Differenz."""

import phn_constants as C
import phn_results as RES

SE_FACTOR = C.SD_FACTOR


def _paired(c, w, pair):
    return c[w]["paired"][pair]


def _r5(c, w, ranker):
    return RES.stat(c, w, ranker, 5)[0]


def criteria_bekannt(c, w):
    d_o_nb, _ = _paired(c, w, "O-NB")
    d_nb_ic, _ = _paired(c, w, "NB-IC")
    return [("Naive Bayes gleich Orakel (Differenz unter 0,05 Punkten)", abs(d_o_nb) < 0.05),
            ("Naive Bayes liegt mindestens 10 Punkte vor Häufigkeit × Informationsgehalt", d_nb_ic >= 10.0)]


def criteria_drei_faelle(c, w):
    d_nb_ic, se = _paired(c, w, "NB-IC")
    return [("Naive Bayes erreicht höchstens 65 Prozent des Orakels", _r5(c, w, "NB") <= 0.65 * _r5(c, w, "O")),
            ("Häufigkeit × Informationsgehalt liegt mehr als zwei Standardfehler vor Naive Bayes", d_nb_ic < -SE_FACTOR * se)]


def criteria_ein_fall(c, w):
    d_nb_ic, _ = _paired(c, w, "NB-IC")
    return [("Häufigkeit × Informationsgehalt liegt mindestens 8 Punkte vor Naive Bayes", d_nb_ic <= -8.0),
            ("Zahl Übereinstimmungen fällt unter 5 Prozent (alle Krankheiten gleich bewertet)", _r5(c, w, "OV") < 5.0)]


def criteria_nicht_geprueft(c, w):
    d, se = _paired(c, w, "NB-NBp")
    return [("„nur dokumentierte Merkmale“ liegt mehr als zwei Standardfehler vor Naive Bayes", d < -SE_FACTOR * se)]


def criteria_viele_luecken(c, w):
    return [("Recall@5 des Orakels höchstens 55 Prozent", _r5(c, w, "O") <= 55.0)]


def criteria_falsch_dokumentiert(c, w):
    d_o_nb, _ = _paired(c, w, "O-NB")
    d_nb_nbp, _ = _paired(c, w, "NB-NBp")
    return [("Naive Bayes liegt höchstens 3 Punkte unter dem Orakel", d_o_nb <= 3.0),
            ("Naive Bayes liegt mindestens 5 Punkte vor „nur dokumentierte Merkmale“", d_nb_nbp >= 5.0)]


CRITERIA = {
    "Wissensbasis bekannt": criteria_bekannt,
    "Seltene Krankheit (3 Fälle)": criteria_drei_faelle,
    "Nur ein Fall": criteria_ein_fall,
    "Nicht alles geprüft": criteria_nicht_geprueft,
    "Viele Lücken": criteria_viele_luecken,
    "Falsch dokumentiert": criteria_falsch_dokumentiert,
}


def check_preset(name: str, c: dict) -> list:
    """Alle Kriterien des Presets an einer Zelle: Liste (Text, Menge, erfüllt) über die Mengen A und B."""
    return [(text, w, bool(ok)) for w in ("A", "B") for text, ok in CRITERIA[name](c, w)]


def check_all() -> dict:
    """Preset -> Liste der Kriterien; die Zelle ist die Einstellung des Presets (ohne Zufalls-Seed)."""
    out = {}
    for name, preset in C.PRESETS.items():
        s = RES.Settings(**{k: v for k, v in preset.items() if k != "seed"})
        out[name] = check_preset(name, RES.cell(s))
    return out
