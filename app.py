"""Phänotyp-Ranking – welche seltene Krankheit passt? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lehrmodell mit erfundenen Daten, keine medizinische Beratung. Aufbau nach DEMO-PLAYBOOK §2 (Fall-Demo).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import phn_constants as C
import phn_evaluation as ev
import phn_live as L
import phn_pdf_export as PDF
import phn_presets as P
import phn_results as RES
import phn_seltene as S
import phn_ui_panel as UI
import phn_visualization as V
from phn_layout import dunkles_design, kopf_css, tabellen_css

st.set_page_config(page_title="Phänotyp-Ranking – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Messe fünf Welten ...")
def _measure(settings):
    return L.measure(settings)


@st.cache_data(show_spinner=False)
def _prevalence(seed, prevalence, n_neg, reps):
    return S.prevalence_row(seed, prevalence, n_neg, reps)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in phn_layout.py).
st.html(kopf_css(dunkles_design()))
st.html(tabellen_css())
with st.container(key="kopf"):
    st.title("🧩 Phänotyp-Ranking – welche seltene Krankheit passt?")
st.markdown(
    """
Aus wenigen **dokumentierten Merkmalen** (Symptomen) eine Rangliste möglicher **seltener Krankheiten** erstellen: Wie gut ist sie, und woran hängt sie?
Die Demo vergleicht Ranker mit dem **Orakel**, dem exakten Posterior des Erzeugers dieser Welt, und zeigt drei Hebel: die **Wissensbasis** (wie viele Fälle je Krankheit bekannt sind),
**Dokumentationslücken** (nicht aufgeschrieben, nicht geprüft, falsch aufgeschrieben) und die **Zahl der Merkmale** je Patient.

**Lehrmodell mit erfundenen Daten, keine medizinische Beratung.** Alle Aussagen gelten für diese Welt, nicht für echte Krankheiten.

Der Expander **💬 In einfachen Worten** erzählt die Idee ohne Formeln. Weiter unten: drei **Kernfragen** und der Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/phaenotyp-ranking-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Bestimmungsbuch für seltene Pilze nennt nach einigen notierten Merkmalen eine Rangliste möglicher Arten; die Demo tut dasselbe mit erfundenen seltenen Krankheiten. Wie gut die Rangliste ist, hängt an drei Dingen: wie viele Fälle hinter jedem Eintrag stehen, ob ein leeres Feld „nicht geprüft“ oder „nicht vorhanden“ heißt und wie viele Merkmale ein Fall überhaupt hat. Stehen hinter einem Eintrag nur wenige Fälle, schätzt das genaue Verfahren die Häufigkeiten zu grob, und eine einfache Regel kann vorn liegen. Wird nur ein Teil der Merkmale abgefragt, darf Fehlendes nicht als abwesend zählen. Die Demo zeigt das an einer **Wissensbasis** aus wenigen Fällen, vergleicht sie mit dem **Orakel**, dem besten Verfahren dieser erfundenen Welt, und wertet bei seltener Klasse aus. Alle Daten sind erfunden; es gibt keine medizinische Beratung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/phaenotyp-ranking-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()


def _stufe(label_text, state_key, help_text, **kw):
    return st.select_slider(label_text, options=P.options(state_key), key=state_key, format_func=lambda v, k=state_key: P.label(k, v), help=help_text, **kw)


with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wissensbasis**")
    nd = _stufe("Zahl der Krankheiten", "nd_select", "Wie viele Krankheiten im Ranking stehen. Mehr Krankheiten heißt mehr Verwechslungen.")
    kb = _stufe("Wissensbasis: Fälle je Krankheit", "kb_select", "bekannt: die wahren Häufigkeiten. Sonst werden die Häufigkeiten aus so vielen gezogenen Fällen je Krankheit geschätzt (selten = wenige Fälle).")
    st.markdown("**Dokumentation**")
    q = _stufe("Dokumentationsverlust q", "q_select", "Anteil der vorhandenen Merkmale, der nicht aufgeschrieben wird.")
    a = _stufe("Geprüfte Merkmale (Abdeckung)", "a_select", "Anteil der Merkmale, die überhaupt abgefragt werden. Was nicht geprüft wurde, steht als „nicht dokumentiert“.")
    fp = _stufe("Falsche Einträge", "fp_select", "Wahrscheinlichkeit, mit der ein nicht vorhandenes Merkmal fälschlich in der Akte steht. Die Wissensbasis kennt diese Fehler nicht, das Orakel schon.")
    st.markdown("**Vorab**")
    pr = _stufe("Vorab-Wahrscheinlichkeit der Krankheiten", "pr_select", "gleichverteilt (Spezialzentrum) oder nach Häufigkeit (Zipf-Verteilung, wenige häufige und viele seltene Krankheiten).")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Messung", *P.bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welten würfeln", width="stretch", on_click=P.randomize_seed)

P.sync_query_params({"nd_select": nd, "kb_select": kb, "q_select": q, "a_select": a, "fp_select": fp, "pr_select": pr, "seed_input": int(seed)})

settings = L.Settings(nd=int(nd), kb=int(kb), q=float(q), a=float(a), fp=float(fp), pr=int(pr), seed=int(seed))
m = _measure(settings)
kz = ev.kennzahlen(m)
few = L.share_few(m)
(v1_kind, v1_text), (v2_kind, v2_text) = ev.verdicts(settings, kz)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eine Messung (live)")
st.caption(f"Diese Messung: {len(m.runs)} Welten × {m.n_patients} Patienten (Zufalls-Seed {settings.seed}), Mittel ± Streuung über die Welten. "
           "Ein einzelner Patient belegt nichts. Die Verdikte unter den Kernfragen 1 und 2 rechnen live aus dieser Messung (gepaart über die Welten); die Abbildungen und Tabellen der Kernfragen stammen aus der vorgerechneten Messreihe.")
UI.metrics_row(settings, kz, few)
st.plotly_chart(V.fig_recall(kz["summary"]), width="stretch", key="main_chart", config=V.CONFIG)
st.caption("Recall@k: Anteil der gerankten Patienten (mit mindestens einem dokumentierten Merkmal), bei denen die wahre Krankheit unter den ersten k der Rangliste steht (Gleichstände zufällig aufgelöst). "
           "Das Orakel ist der exakte Posterior dieser Welt, eine Obergrenze nur innerhalb dieses Modells.")

if st.button("📄 PDF der Messung erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = PDF.build_pdf(settings, kz, (( v1_kind, v1_text), (v2_kind, v2_text)), len(m.runs), m.n_patients)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="phaenotyp-ranking-messung.pdf", mime="application/pdf", key="pdf_download")

st.markdown("### 🧍 Ein Beispielpatient")
UI.patient_panel(m, settings, len(m.runs[0].example["d"]))

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

meta = RES.load()["meta"]
vorgerechnet = (f"Vorgerechnet: {meta['cells']} Zellen, je zwei Mengen von {meta['worlds_per_set']} Welten (Zufalls-Seed {meta['sets']['A']} und {meta['sets']['B']}), "
                f"{meta['patients_per_world']} Patienten je Welt; sonst Voreinstellung.")

st.markdown("---")
st.markdown("## ❓ Kernfrage 1: Wie viel Wissen braucht die Rangliste?")
getattr(st, v1_kind)(v1_text)
st.plotly_chart(V.fig_kb_curve(settings.kb), width="stretch", key="kb_chart", config=V.CONFIG)
st.caption(vorgerechnet + " Die Pseudozählung der Schrumpfung ist gesetzt, nicht abgestimmt.")

st.markdown("## ❓ Kernfrage 2: Ist ein nicht dokumentiertes Merkmal abwesend?")
getattr(st, v2_kind)(v2_text)
st.plotly_chart(V.fig_coverage(), width="stretch", key="coverage_chart", config=V.CONFIG)
st.caption(vorgerechnet)
by = L.by_nobs(m)
st.plotly_chart(V.fig_nobs(by), width="stretch", key="nobs_chart", config=V.CONFIG)
st.caption(f"In dieser Messung haben {few:.0f} % der Patienten höchstens zwei dokumentierte Merkmale; ein einzelner mittlerer Recall verdeckt das.")

sel = RES.load()["seltene"]
rows = sel["prevalence"]
st.markdown("## ❓ Kernfrage 3: Wie wertet man bei wenigen Positiven aus?")
k3_kind, k3_text = ev.verdict_seltene(rows[0]["auc"][0], rows[-1]["auc"][0], rows[0]["ap"][0], rows[-1]["ap"][0], rows[0]["prevalence"], rows[-1]["prevalence"])
getattr(st, k3_kind)(k3_text)
c1, c2 = st.columns(2)
c1.plotly_chart(V.fig_auc_ap(rows), width="stretch", key="auc_ap_chart", config=V.CONFIG)
w_rows = ["| Positive | Wald überdeckt | Wilson überdeckt |", "|---|---|---|"]
w_rows += [f"| {w['n_pos']} | {ev.dec(100 * w['wald'], 1)} % | {ev.dec(100 * w['wilson'], 1)} % |" for w in sel["wilson"]]
s_rows = ["| Positive | Streuung der AUC | Abweichung über 0,03 |", "|---|---|---|"]
s_rows += [f"| {w['n_pos']} | {ev.dec(w['sd'])} | {ev.dec(w['share_off'], 1)} % |" for w in sel["spread"]]
c2.markdown("**95-%-Intervall für eine Sensitivität von 0,9** (Überdeckung, " + f"{sel['wilson'][0]['reps']:,}".replace(",", ".") + " Wiederholungen):\n\n" + "\n".join(w_rows))
c2.markdown("**Streuung der AUC-Schätzung** (Prävalenz 1 %, wahre AUC " + ev.dec(sel["true_auc"], 2) + f", {sel['spread'][0]['reps']} Datensätze):\n\n" + "\n".join(s_rows))
st.caption("Scores: Negative gleichverteilt, Positive mit Potenz 1/19 (wahre AUC 0,95, ohne Normalverteilung). Voraussetzungen und Zahl der Ziehungen im Tab „🎯 Seltene Klasse“.")

st.markdown("---")

# --- Methodenvergleich ---------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_r, tab_w, tab_n, tab_k, tab_s = st.tabs(["📏 Ranker", "📚 Wissensbasis", "🔍 Nicht geprüft", "📐 Kalibrierung", "🎯 Seltene Klasse"])
    with tab_r:
        st.markdown(UI.ranker_table(kz["summary"]))
        st.markdown(
            "- **Orakel:** exakter Posterior des Erzeugers; kennt die wahren Häufigkeiten, das wahre Vorab, die Abdeckung und die falschen Einträge.\n"
            "- **Naive Bayes (Add-½-Glättung):** log P(d) plus für jedes Merkmal log g beziehungsweise log (1 − g); nicht Dokumentiertes zählt als abwesend; die Häufigkeiten g sind mit Add-½-Glättung geschätzt (zu jeder Zählung wird 0,5 addiert).\n"
            "- **Naive Bayes (Schrumpfung 3 und 10):** dasselbe mit Häufigkeiten, die zum Merkmalsmittel geschrumpft sind (Pseudozählung 3 beziehungsweise 10); bei bekannter Wissensbasis gleich der Variante mit Add-½-Glättung.\n"
            "- **Naive Bayes ohne Vorab:** die Variante mit Add-½-Glättung ohne den Vorab-Term.\n"
            "- **Nur dokumentierte Merkmale:** nur dokumentierte Merkmale zählen, Abwesenheit ist keine Evidenz.\n"
            "- **Häufigkeit × Informationsgehalt:** Summe über dokumentierte Merkmale von Häufigkeit mal Informationsgehalt (seltene Merkmale zählen mehr).\n"
            "- **Zahl Übereinstimmungen:** wie viele dokumentierte Merkmale das Profil überhaupt kennt (Häufigkeit über 0,05); bei Gleichstand zufällige Reihenfolge."
        )
        st.caption("Streuung = Standardabweichung der Mittel der einzelnen Welten (n − 1). Gleichstände werden als Wahrscheinlichkeit gezählt, nicht als Treffer oder Fehlschlag.")
    with tab_w:
        st.markdown(f"Aktuelle Wissensbasis: **{P.label('kb_select', settings.kb)}**." + ("" if settings.kb > 0 else " Bei bekannter Wissensbasis sind die drei Naive-Bayes-Varianten gleich."))
        cells = RES.load()["cells"]
        spalten = ("O", "NB", "NS3", "NS10", "NBp", "IC", "OV")
        rows_w = ["| Fälle je Krankheit | " + " | ".join(C.RANKER_NAMES[r] for r in spalten) + " |", "|---|" + "---|" * len(spalten)]
        for kb_v in (1, 3, 10, 30):
            c = cells[RES.Settings(kb=kb_v).key()]
            rows_w.append(f"| {kb_v} | " + " | ".join(ev.pct(RES.pooled(c, r)) for r in spalten) + " |")
        c = cells[RES.BASE.key()]
        rows_w.append("| bekannt | " + " | ".join(ev.pct(RES.pooled(c, r)) for r in spalten) + " |")
        st.markdown("Recall@5 in %, vorgerechnet (Mittel der beiden Mengen, zehn Welten):\n\n" + "\n".join(rows_w))
        st.caption("Die drei Naive-Bayes-Varianten nutzen dieselben Fälle und unterscheiden sich nur in der Schätzung der Häufigkeiten: Add-½-Glättung (0,5 je Zählung) oder Schrumpfung zum Merkmalsmittel mit Pseudozählung 3 beziehungsweise 10 "
                   "(gesetzt, nicht abgestimmt). Bei 1 und 3 Fällen liegt mit Add-½ eine einfache Regel vor Naive Bayes, mit Schrumpfung liegt Naive Bayes vor ihr; bei 1 und 3 Fällen ist die Schrumpfung 3 besser als Add-½, "
                   "bei 30 Fällen ist Add-½ besser als Schrumpfung 3. Keine Schätzung gewinnt überall.")
    with tab_n:
        st.markdown("**Nicht geprüft ist nicht abwesend.** Naive Bayes liest jedes Merkmal, das nicht in der Liste steht, als abwesend. Wurde es aber nie abgefragt, ist das ein Fehlschluss; "
                    "das Orakel kennt die Maske der geprüften Merkmale und lässt nicht Geprüftes aus.")
        st.plotly_chart(V.fig_coverage(), width="stretch", key="coverage_chart_tab", config=V.CONFIG)
        cells = RES.load()["cells"]
        rows_f = ["| Falsche Einträge | Orakel | Naive Bayes | nur dokumentierte |", "|---|---|---|---|"]
        for fp_v in C.FP_OPTIONS:
            c = cells[RES.Settings(fp=fp_v).key()]
            rows_f.append(f"| {P.label('fp_select', fp_v)} | " + " | ".join(ev.pct(RES.pooled(c, r)) for r in ("O", "NB", "NBp")) + " |")
        st.markdown("Falsch dokumentierte Merkmale (Recall@5 in %, vorgerechnet, Wissensbasis bekannt):\n\n" + "\n".join(rows_f))
    with tab_k:
        ranker_cal = st.radio("Ranker", ["NB", "O"], format_func=lambda r: C.RANKER_NAMES[r], horizontal=True, key="calib_ranker")
        cal = L.calibration(m, ranker_cal)
        st.plotly_chart(V.fig_calibration(cal["bins"], C.RANKER_NAMES[ranker_cal]), width="stretch", key=f"calib_chart_{ranker_cal}", config=V.CONFIG)
        st.markdown(f"Mittlere genannte Wahrscheinlichkeit des Spitzenreiters: **{ev.pct(cal['mean_p'])} %**, tatsächliche Trefferquote: **{ev.pct(cal['hit'])} %** (diese Messung).")
        st.caption("Eine Rangliste ist keine Wahrscheinlichkeit: Nur wenn die Wissensbasis stimmt, alle Merkmale geprüft wurden und keine falschen Einträge vorkommen, passen genannte Wahrscheinlichkeit "
                   "und Trefferquote zusammen (bei Abdeckung 0,2 und bei fünf Prozent falschen Einträgen nennt Naive Bayes in dieser Welt deutlich mehr, als es trifft). Bei geschätzter Wissensbasis "
                   "kann die Abweichung in beide Richtungen gehen; sie hängt an der Schätzung der Häufigkeiten (bei 3 Fällen: Add-½ und Schrumpfung 10 nennen weniger, als sie treffen, Schrumpfung 3 mehr; gemessen, siehe README).")
    with tab_s:
        n_neg_text = f"{C.LIVE_N_NEG:,}".replace(",", ".")
        st.markdown(f"Gleiche Trennschärfe, andere Mischung der Klassen: eine **Prävalenz** wählen und live nachrechnen ({C.LIVE_REPS} Ziehungen, {n_neg_text} Negative; vorgerechnet mit mehr Ziehungen siehe oben).")
        prev = st.select_slider("Anteil der Positiven (Prävalenz)", options=[0.5, 0.1, 0.01, 0.001], value=0.01, key="prev_select",
                                format_func=lambda v: f"{100 * v:g} %".replace(".", ","))
        row = _prevalence(settings.seed, prev, C.LIVE_N_NEG, C.LIVE_REPS)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Positive", f"{row['n_pos']}")
        r2.metric("ROC-AUC", ev.dec(row["auc"][0]), delta=f"± {ev.dec(row['auc'][1])}", delta_color="off")
        r3.metric("Average Precision", ev.dec(row["ap"][0]), delta=f"± {ev.dec(row['ap'][1])}", delta_color="off")
        r4.metric("PPV bei Sensitivität 90 %", ev.dec(row["ppv"][0]), delta=f"± {ev.dec(row['ppv'][1])}", delta_color="off")
        st.caption(f"Live: {row['reps']} Ziehungen, {row['n_neg']} Negative; Mittel ± Streuung über die Ziehungen. Die AUC ändert sich mit der Prävalenz nicht, die Average Precision schon: "
                   "wer bei seltener Klasse nur die AUC berichtet, verdeckt das.")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        f"""
**Die Welt.** {C.NS} Merkmale mit Zipf-verteilter Beliebtheit: beliebte Merkmale kommen in vielen Krankheitsprofilen und im Hintergrund vor und tragen wenig Information. Jede Krankheit hat
{C.K_MIN_FEAT} bis {C.K_MAX_FEAT} Profilmerkmale mit Häufigkeit 0,95, 0,6, 0,3 oder 0,1. Ein vorhandenes Merkmal wird mit Wahrscheinlichkeit 1 − q aufgeschrieben, dazu kommt eine Hintergrundrate je Merkmal.

**Patienten.** Krankheit nach dem Vorab, Merkmale unabhängig gegeben die Krankheit. Patienten ohne jedes dokumentierte Merkmal werden nicht gerankt: in den vorgerechneten Zellen bleiben 97,8 Prozent der Patienten in der Voreinstellung, 93,9 Prozent bei Dokumentationsverlust 0,6, 84,9 Prozent bei Abdeckung 0,5 und 53,4 Prozent bei Abdeckung 0,2; die Recall-Werte gelten für die gerankten Patienten. Mit Abdeckung unter 1 wird nur ein Teil der Merkmale geprüft.

**Wissensbasis.** Entweder die wahren Häufigkeiten, oder Häufigkeiten aus gezogenen Fällen je Krankheit. Aus denselben Fällen entstehen drei Schätzungen: Add-½-Glättung sowie Schrumpfung zum Merkmalsmittel mit Pseudozählung 3 und 10; sie speisen drei Naive-Bayes-Varianten nebeneinander.

**Messung.** Je Einstellung {C.N_WORLDS} Welten mit je {C.N_PATIENTS} Patienten (SplitMix64: der Zufallsstrom ist auf jeder Plattform derselbe; Gleitkomma-Rundung kann die Recall-Werte um unter 0,6 Punkte verschieben). Der Recall ist ein Erwartungswert über die zufällige Auflösung von Gleichständen.

**Auswertung gepaart.** Ob ein Ranker vor einem anderen liegt, entscheidet der Mittelwert der Differenzen über die Welten gegen seinen Standardfehler (zwei Standardfehler als Schwelle).
"""
    )
    st.markdown("#### 🚧 Wo die Annahmen enden")
    st.markdown(
        "| Annahme der Demo | Was in der Wirklichkeit anders ist |\n|---|---|\n"
        "| Merkmale unabhängig gegeben die Krankheit | echte Merkmale sind abhängig und haben Hierarchien (Ontologie) |\n"
        "| Das Orakel ist der exakte Posterior des Erzeugers | in echten Daten gibt es kein Orakel |\n"
        "| 50 bis 1000 Krankheiten, 100 Merkmale | echte Wissensbasen haben mehrere Tausend Krankheiten und Merkmale |\n"
        "| Merkmale nur an oder aus | Schweregrad, Alter, Verlauf fehlen |\n"
        "| Häufigkeiten aus gezogenen Fällen derselben Welt | echte Berichte sind verzerrt und uneinheitlich |\n"
        "| Falsche Einträge als gleichmäßige Rate | echte Fehler häufen sich bei bestimmten Merkmalen |"
    )

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Erfundene Welt mit Naive-Bayes-Struktur (Merkmale unabhängig gegeben die Krankheit). Parameter (Häufigkeiten, Hintergrund, Dokumentationsverlust) sind gesetzt, nicht aus Daten geschätzt.
Die Wissensbasis wird aus Fällen derselben Welt geschätzt.

**Grenzen.** Das Orakel gehört zum Erzeuger und ist nur in diesem Modell eine Obergrenze. Fünf Welten je Messung: die Streuung über die Welten ist ausgewiesen, Unterschiede unter zwei Standardfehlern werden nicht gedeutet.
Die Schätzungen (Add-½-Glättung, Schrumpfung 3 und 10) sind gesetzt und nicht abgestimmt. Der Abschnitt „seltene Klasse“ nutzt Scores mit bekannter AUC, keine Ranker aus dieser Welt.

**Bekannte Probleme und Unsicherheiten.** Nur an erfundenen Welten geprüft; keine echten Patientendaten, keine echte Wissensbasis (HPO, Orphanet), keine Ontologie. Die Aussagen gelten für die gemessenen Zellen
und die gemessene Regel jedes Rankers, nicht für die Methodenklasse. **Keine medizinische Beratung, keine Diagnoseempfehlung.**
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/phaenotyp-ranking-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben: die dokumentierten Merkmale $x$ eines Patienten, die Wissensbasis (Häufigkeiten $g$) und das Vorab $\pi$. Gesucht: eine Rangliste der Krankheiten $d$.

| Symbol | Bedeutung |
|---|---|
| $d$, $s$ | Krankheit, Merkmal |
| $x_s \in \{0,1\}$ | Merkmal $s$ ist dokumentiert |
| $z_s \in \{0,1\}$ | Merkmal $s$ wurde geprüft |
| $\pi_d$ | Vorab der Krankheit $d$ |
| $f_{ds}$ | Häufigkeit des Profilmerkmals bei Krankheit $d$ |
| $q$ | Dokumentationsverlust |
| $b_s$ | Hintergrundrate des Merkmals $s$ |
| $g_{ds}$ | Wahrscheinlichkeit, dass $s$ bei Krankheit $d$ dokumentiert wird |
| $g'_{ds}$ | $g_{ds}$ mit falschen Einträgen: $1 - (1 - g_{ds})(1 - r)$ (nur das Orakel kennt sie) |
| $r$ | Rate falscher Einträge (0 / 0,02 / 0,05) |
| $\hat g_{ds}$ | aus $n$ Fällen geschätzte Häufigkeit |
| $c_{ds}$ | Zahl der Fälle (von $n$), in denen das Merkmal bei Krankheit $d$ dokumentiert war |
| $\bar m_s$ | Merkmalsmittel: Mittel der Anteile $c_{ds}/n$ über alle Krankheiten, mindestens 0,001 |
| $a$ | Pseudozählung der Schrumpfung (3 oder 10) |
| $\alpha$ | Abdeckung: Anteil der geprüften Merkmale |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/phaenotyp-ranking-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown("Erzeuger (Dokumentationswahrscheinlichkeit, mit $b_s = 0{,}01 + 0{,}2\\,(w_s / w_{\\max})^2$, $w_s \\propto \\mathrm{Rang}^{-0{,}8}$):")
    st.latex(r"g_{ds} = 1 - \bigl(1 - f_{ds}(1 - q)\bigr)\bigl(1 - b_s\bigr)")
    st.markdown("Orakel (exakter Posterior, nur geprüfte Merkmale, $z_s = 1$; mit der Rate $r$ falscher Einträge steht $g'_{ds} = 1 - (1 - g_{ds})(1 - r)$, in der Voreinstellung ist $r = 0$ und $g' = g$):")
    st.latex(r"P(d \mid x, z) \;\propto\; \pi_d \prod_{s:\, z_s = 1} {g'_{ds}}^{\,x_s}\,\bigl(1 - g'_{ds}\bigr)^{1 - x_s}")
    st.markdown("Naive Bayes (alle Merkmale zählen, Nicht-Dokumentiertes als abwesend, mit geschätzten $\\hat g$):")
    st.latex(r"\log P(d \mid x) = \log \pi_d + \sum_{s} \bigl[x_s \log \hat g_{ds} + (1 - x_s)\log(1 - \hat g_{ds})\bigr] + \text{const.}")
    st.markdown("Schätzung aus $n$ Fällen (Zahl $c_{ds}$ der Fälle mit dokumentiertem Merkmal): Add-½-Glättung (zu jeder Zählung wird 0,5 addiert, nicht die klassische Laplace-Glättung) und Schrumpfung zum Merkmalsmittel $\\bar m_s$ (Mittel der Anteile $c_{ds}/n$ über alle Krankheiten, mindestens 0,001) mit Pseudozählung $a$:")
    st.latex(r"\hat g_{ds} = \frac{c_{ds} + 0{,}5}{n + 1}, \qquad \hat g_{ds} = \frac{c_{ds} + a\,\bar m_s}{n + a}")
    st.markdown("Häufigkeit × Informationsgehalt ($\\bar g_s$ = Mittel von $\\hat g_{ds}$ über alle Krankheiten; der Informationsgehalt wird um sein Minimum verschoben, sodass der kleinste Wert 0,001 ist):")
    st.latex(r"\mathrm{IC}_s = -\ln \bar g_s - \min_{s'}\bigl(-\ln \bar g_{s'}\bigr) + 0{,}001, \qquad \mathrm{score}(d) = \sum_s x_s\,\hat g_{ds}\,\mathrm{IC}_s")
    st.markdown("Recall@$k$ mit zufälliger Auflösung von Gleichständen ($r_i$ kleinster Rang der wahren Krankheit, $t_i$ Zahl der Gleichstände mit ihr):")
    st.latex(r"\mathrm{Recall@}k = \frac{1}{N}\sum_{i=1}^{N} \min\Bigl(1,\; \max\Bigl(0,\; \frac{k - r_i + 1}{t_i + 1}\Bigr)\Bigr)")
    st.caption("Der Formelblock zeigt Erzeuger, Orakel, Naive Bayes, Schätzung, Häufigkeit × Informationsgehalt und Recall. Nicht in den Gleichungen stehen: die Varianten Naive Bayes ohne Vorab, nur dokumentierte Merkmale und Zahl Übereinstimmungen, "
               "die Untergrenze 10⁻⁶ der Häufigkeiten in den Logarithmen und die Zufallsziehungen (die Untergrenze 0,001 des Merkmalsmittels steht im Text neben den Gleichungen) "
               "(Profilmerkmale ohne Zurücklegen proportional zur Beliebtheit, Häufigkeiten 0,95 / 0,6 / 0,3 / 0,1 "
               "mit den Anteilen 15 / 25 / 35 / 25 Prozent); sie stehen im README und im Code.")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
