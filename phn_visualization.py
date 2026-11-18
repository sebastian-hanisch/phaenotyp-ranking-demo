"""Plotly-Abbildungen (alle Achsen gesperrt, transparenter Hintergrund). Zahlen kommen aus der Messung, nie aus dem Text."""

import numpy as np
import plotly.graph_objects as go

import phn_constants as C
import phn_results as RES

CONFIG = {"displayModeBar": False}
_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=20, t=40, b=60), legend=dict(orientation="h", y=-0.25))


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def fig_recall(summary: dict, rankers=C.MAIN_RANKERS) -> go.Figure:
    """Recall@1, @5, @10 je Ranker (Balken) mit Streuung über die Welten (Fehlerbalken)."""
    fig = go.Figure()
    xs = ["Recall@1", "Recall@5", "Recall@10"]
    for r in rankers:
        ys = [summary[r][k][0] for k in (1, 5, 10)]
        err = [summary[r][k][1] for k in (1, 5, 10)]
        fig.add_bar(name=C.RANKER_NAMES[r], x=xs, y=ys, error_y=dict(type="data", array=err, visible=True), marker_color=C.RANKER_COLORS[r],
                    hovertemplate="%{y:.1f} %<extra>" + C.RANKER_NAMES[r] + "</extra>")
    fig.update_layout(barmode="group", yaxis_title="Anteil der Patienten in %", yaxis_range=[0, 105], title="Recall der Ranker (Mittel ± Streuung über die Welten)", **_LAYOUT)
    return _lock(fig)


def fig_kb_curve(current_kb: int) -> go.Figure:
    """Recall@5 über die Größe der Wissensbasis (vorgerechnet, Mittel der beiden Mengen zu je fünf Welten), Voreinstellung sonst; Naive Bayes mit Laplace und mit Schrumpfung 3."""
    cells = RES.load()["cells"]
    xs = [1, 3, 10, 30]
    labels = ["1 Fall", "3 Fälle", "10 Fälle", "30 Fälle", "bekannt"]

    def pool(r):
        out = []
        for kb in xs:
            out.append(RES.pooled(cells[RES.Settings(kb=kb).key()], r))
        out.append(RES.pooled(cells[RES.BASE.key()], r))
        return out

    series = [("O", "Orakel", C.COLOR_O), ("NB", "Naive Bayes (Laplace)", C.COLOR_NB), ("NS3", "Naive Bayes (Schrumpfung 3)", C.RANKER_COLORS["NS3"]),
              ("IC", "Häufigkeit × Informationsgehalt", C.COLOR_IC), ("OV", "Zahl Übereinstimmungen", C.COLOR_OV)]
    fig = go.Figure()
    for r, name, color in series:
        fig.add_scatter(x=labels, y=pool(r), mode="lines+markers", name=name, line=dict(color=color, dash="dot" if r == "O" else "solid"))
    cur = {0: "bekannt", 1: "1 Fall", 3: "3 Fälle", 10: "10 Fälle", 30: "30 Fälle"}[current_kb]
    fig.add_vline(x=labels.index(cur), line_dash="dash", line_color="rgba(128,128,128,0.7)", annotation_text="gewählt", annotation_position="top")
    fig.update_layout(yaxis_title="Recall@5 in %", yaxis_range=[0, 100], title="Recall@5 nach Größe der Wissensbasis (vorgerechnet, 10 Welten)", **_LAYOUT)
    return _lock(fig)


def fig_coverage() -> go.Figure:
    """Recall@5 bei voller, halber und einem Fünftel Abdeckung: Orakel (kennt die Maske), Naive Bayes (nicht dokumentiert = abwesend), nur dokumentierte."""
    cells = RES.load()["cells"]
    xs = ["alle geprüft (1,0)", "die Hälfte (0,5)", "ein Fünftel (0,2)"]
    keys = [RES.Settings(a=1.0).key(), RES.Settings(a=0.5).key(), RES.Settings(a=0.2).key()]
    fig = go.Figure()
    for r, color in (("O", C.COLOR_O), ("NB", C.COLOR_NB), ("NBp", C.COLOR_NBP)):
        fig.add_bar(name={"O": "Orakel (kennt die Maske)", "NB": "Naive Bayes: nicht dokumentiert = abwesend", "NBp": "nur dokumentierte Merkmale"}[r],
                    x=xs, y=[RES.pooled(cells[k], r) for k in keys], marker_color=color)
    fig.update_layout(barmode="group", yaxis_title="Recall@5 in %", yaxis_range=[0, 100], title="Nicht jedes Merkmal wurde geprüft (vorgerechnet, 10 Welten)", **_LAYOUT)
    return _lock(fig)


def fig_nobs(by_nobs: dict, rankers=("O", "NB", "NBp", "IC")) -> go.Figure:
    """Recall@5 nach Zahl der dokumentierten Merkmale, mit dem Anteil der Patienten je Gruppe in der Achsenbeschriftung."""
    labs = [f"{lo}–{hi if hi < 1000 else '…'} Merkmale<br>({s:.0f} % der Patienten)" if hi < 1000 else f"ab {lo} Merkmalen<br>({s:.0f} %)"
            for (lo, hi), s in zip(C.BINS_NOBS, by_nobs["share"])]
    fig = go.Figure()
    for r in rankers:
        fig.add_bar(name=C.RANKER_NAMES[r], x=labs, y=by_nobs["recall5"][r], marker_color=C.RANKER_COLORS[r])
    fig.update_layout(barmode="group", yaxis_title="Recall@5 in %", yaxis_range=[0, 105], title="Recall@5 nach Zahl dokumentierter Merkmale (diese Messung)", **_LAYOUT)
    return _lock(fig)


def fig_calibration(bins: list, label: str) -> go.Figure:
    """Zuverlässigkeitsdiagramm: genannte Spitzen-Wahrscheinlichkeit gegen Trefferquote je Fach (Größe der Punkte = Zahl der Patienten)."""
    pts = [(b[1], b[2], b[0]) for b in bins if b[0] > 0 and not np.isnan(b[1])]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(128,128,128,0.7)", dash="dash"), name="perfekt kalibriert", hoverinfo="skip")
    if pts:
        n = np.array([p[2] for p in pts], dtype=float)
        fig.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+lines", name=label, marker=dict(size=8 + 22 * np.sqrt(n / n.max()), color=C.COLOR_NB),
                        text=[f"{int(v)} Patienten" for v in n], hovertemplate="genannt %{x:.2f}, Treffer %{y:.2f}<br>%{text}<extra></extra>")
    fig.update_layout(xaxis_title="genannte Wahrscheinlichkeit des Spitzenreiters", yaxis_title="Trefferquote", xaxis_range=[0, 1.02], yaxis_range=[0, 1.02],
                      title=f"Kalibrierung: {label}", **_LAYOUT)
    return _lock(fig)


def fig_auc_ap(rows: list) -> go.Figure:
    """AUC und Average Precision über die Prävalenz (gleiche Trennschärfe)."""
    labs = [f"{100 * r['prevalence']:g}".replace(".", ",") + " %" for r in rows]
    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=labs, y=[r["auc"][0] for r in rows], marker_color=C.COLOR_O)
    fig.add_bar(name="Average Precision", x=labs, y=[r["ap"][0] for r in rows], marker_color=C.COLOR_IC)
    fig.update_layout(barmode="group", xaxis_title="Anteil der Positiven (Prävalenz)", yaxis_title="Wert", yaxis_range=[0, 1.05],
                      title="Gleiche Trennschärfe, ganz andere Average Precision", **_LAYOUT)
    return _lock(fig)


def fig_patient(posterior: dict, true_id: int | None) -> go.Figure:
    """Posterior von Naive Bayes für die obersten Krankheiten eines Beispielpatienten; die wahre Krankheit ist hervorgehoben, wenn sie aufgelöst ist."""
    ids = list(posterior)
    colors = [C.COLOR_IC if (true_id is not None and i == true_id) else C.COLOR_NB for i in ids]
    fig = go.Figure(go.Bar(x=[posterior[i] for i in ids], y=[f"Krankheit {i + 1}" for i in ids], orientation="h", marker_color=colors,
                           hovertemplate="%{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Posterior (Naive Bayes)", xaxis_range=[0, 1], yaxis_autorange="reversed", title="Die obersten fünf Krankheiten", **{**_LAYOUT, "legend": None})
    return _lock(fig)
