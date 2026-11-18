"""Darstellung einzelner Bausteine der Seite (Streamlit): Kennzahlen, Beispielpatient, Tabelle der Ranker. Rechnet nichts selbst."""

import streamlit as st

import phn_constants as C
import phn_evaluation as ev
import phn_visualization as V


def metrics_row(settings, k: dict, few: float):
    """Vier Kennzahlen im 2 x 2-Raster; Deltas immer „mein Wert minus Referenz“, höher ist besser."""
    s = k["summary"]
    o5, nb5 = s["O"][5][0], s["NB"][5][0]
    name = C.RANKER_NAMES[k["simple"]]
    sv = k["simple_val"][0]
    c1, c2 = st.columns(2)
    c1.metric("Recall@5 des Orakels", f"{ev.pct(o5)} %", delta="Obergrenze in diesem Modell", delta_color="off")
    c2.metric("Recall@5 Naive Bayes", f"{ev.pct(nb5)} %", delta=f"{ev.spct(nb5 - o5)} Punkte gegen das Orakel", delta_color="normal")
    c3, c4 = st.columns(2)
    c3.metric(f"Beste einfache Regel: {name}", f"{ev.pct(sv)} %", delta=f"{ev.spct(sv - nb5)} Punkte gegen Naive Bayes", delta_color="normal")
    c4.metric("Patienten mit höchstens zwei Merkmalen", f"{few:.0f} %", delta="Anteil der Patienten", delta_color="off")


def ranker_table(summary: dict) -> str:
    """Markdown-Tabelle: Recall@1, @5, @10 und reziproker mittlerer Rang je Ranker (Mittel ± Streuung über die Welten, Prozent beziehungsweise Prozent des reziproken Rangs)."""
    rows = ["| Ranker | Recall@1 | Recall@5 | Recall@10 | reziproker mittlerer Rang |", "|---|---|---|---|---|"]
    for r in C.RANKERS:
        cells = [f"{ev.pct(summary[r][k][0])} ± {ev.pct(summary[r][k][1])}" for k in (1, 5, 10, "mrr")]
        rows.append(f"| {C.RANKER_NAMES[r]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def patient_panel(m, settings, n_example: int):
    """Beispielpatient: dokumentierte Merkmale, Rangliste der obersten fünf Krankheiten je Ranker, Posterior von Naive Bayes; die wahre Krankheit erst nach „Auflösen“."""
    idx = st.slider("Patient", 1, n_example, 1, key="patient_idx", help="Die ersten Patienten der ersten Welt.") - 1
    reveal = st.toggle("Wahre Krankheit zeigen", key="patient_reveal", value=False)
    v = ev.example_view(m, idx)
    feats = ", ".join(f"M{i + 1}" for i in v["features"])
    st.markdown(f"**Dokumentierte Merkmale ({len(v['features'])}):** {feats}")
    if settings.a < 1.0:
        st.caption(f"Geprüft wurden {v['checked']} von {C.NS} Merkmalen; nicht Geprüftes steht nicht in der Liste, ist aber nicht abwesend.")
    true_id = v["true"] if reveal else None
    rows = ["| Ranker | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
    for r in C.RANKERS:
        ids = v["top"][r]["ids"]
        cells = [f"K{i + 1}" + (" ✅" if true_id is not None and i == true_id else "") for i in ids]
        rows.append(f"| {C.RANKER_NAMES[r]} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(rows))
    if reveal:
        ranks = ", ".join(f"{C.RANKER_NAMES[r]}: Rang {v['top'][r]['rank_true']}" for r in C.RANKERS)
        st.caption(f"Wahre Krankheit: K{v['true'] + 1}. Ihr Rang je Ranker: {ranks}.")
    st.plotly_chart(V.fig_patient(v["posterior_nb"], true_id), width="stretch", key=f"patient_chart_{idx}_{reveal}", config=V.CONFIG)
