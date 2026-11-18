"""Konstanten: Regler-Stufen, feste Annahmen der Welt, Farben und Presets (Phänotyp-Ranking, Lehrmodell mit erfundenen Daten)."""

# --- Welt (Annahmen des Erzeugers) -------------------------------------------------------------------------------------------
NS = 100                      # Zahl der Merkmale (Symptome)
POP_EXP = 0.8                 # Zipf-Exponent der Beliebtheit der Merkmale: beliebte Merkmale kommen in vielen Profilen und im Hintergrund vor
FREQ = (0.95, 0.6, 0.3, 0.1)  # Häufigkeit eines Profilmerkmals bei einer Person mit der Krankheit
FREQ_P = (0.15, 0.25, 0.35, 0.25)  # Anteile dieser Häufigkeiten an den Profilmerkmalen
K_MIN_FEAT, K_MAX_FEAT = 4, 12     # Zahl der Profilmerkmale je Krankheit (gleichverteilt, beide Grenzen inklusive)
BG_BASE, BG_SLOPE, BG_POWER = 0.01, 0.2, 2   # Hintergrundrate eines Merkmals = 0,01 + 0,2 * (Beliebtheit / größte Beliebtheit)^2
PRIOR_ZIPF_EXP = 1.0          # Vorab "nach Häufigkeit": Zipf-Exponent 1
LAPLACE_A = 0.5               # Glättung je Seite bei geschätzter Wissensbasis
PROB_FLOOR = 1e-6             # Untergrenze der Häufigkeiten im Logarithmus
SIMPLE_THRESHOLD = 0.05       # "Zahl Übereinstimmungen" zählt Merkmale mit geschätzter Häufigkeit über dieser Schwelle

# --- Messung ------------------------------------------------------------------------------------------------------------------------------
N_WORLDS = 5                  # Wissensbasen (Welten) je Messung
LIVE_N_NEG, LIVE_REPS = 20000, 20   # Tab „Seltene Klasse“: Negative und Ziehungen der Live-Rechnung
N_PATIENTS = 1500             # Patienten je Welt (vor dem Entfernen von Patienten ohne dokumentiertes Merkmal)
SWEEP_SEEDS = {"A": 1000, "B": 2000}   # zwei Mengen von je N_WORLDS Welten (Zufalls-Seed der Messung); nie ein einzelner Seed als Befund
RANKERS = ("O", "NB", "NS3", "NS10", "NB0", "NBp", "IC", "OV")
MAIN_RANKERS = ("O", "NB", "NS3", "NBp", "IC", "OV")   # in den Abbildungen; die Tabellen zeigen alle
SHRINK_A = (3, 10)            # Pseudozählungen der Schrumpfung zum Merkmalsmittel (gesetzt, nicht abgestimmt)
SIMPLE_RANKERS = ("IC", "OV")  # "einfache Regeln" im Sinn der Texte
RANKER_NAMES = {
    "O": "Orakel (exakter Posterior)",
    "NB": "Naive Bayes (Laplace-Glättung)",
    "NS3": "Naive Bayes (Schrumpfung 3)",
    "NS10": "Naive Bayes (Schrumpfung 10)",
    "NB0": "Naive Bayes ohne Vorab",
    "NBp": "nur dokumentierte Merkmale",
    "IC": "Häufigkeit × Informationsgehalt",
    "OV": "Zahl Übereinstimmungen",
}
BINS_NOBS = ((1, 2), (3, 4), (5, 7), (8, 10_000))
CALIB_EDGES = (0.0, 0.5, 0.9, 0.99, 1.0000001)
VERDICT_RATIO_OK = 0.90       # Naive Bayes erreicht mindestens diesen Anteil des Orakels: "die Wissensbasis reicht"
VERDICT_RATIO_THIN = 0.50     # darunter: "zu dünn"
SD_FACTOR = 2.0               # Unterschiede unter dieser Zahl von Streuungen über die Welten werden nicht gedeutet

# --- Regler (Stufen = gemessene Zellen) ---------------------------------------------------------------------------------------------------
ND_OPTIONS = (50, 200, 1000)
KB_OPTIONS = (0, 1, 3, 10, 30)       # 0 = Wissensbasis bekannt, sonst Fälle je Krankheit
Q_OPTIONS = (0.0, 0.3, 0.6)
A_OPTIONS = (1.0, 0.5, 0.2)
FP_OPTIONS = (0.0, 0.02, 0.05)
PR_OPTIONS = (0, 1)                  # 0 = Vorab gleichverteilt, 1 = Vorab nach Häufigkeit (Zipf)
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT = {"nd": 200, "kb": 0, "q": 0.3, "a": 1.0, "fp": 0.0, "pr": 0, "seed": DEFAULT_SEED}
# Rechenaufwand am Regler-Maximum (ND 1000, 1500 Patienten, fünf Welten): Matrizen 1500 x 1000 je Ranker, Profil-Schätzung
# nd * n_case * NS = 3 Mio. Ziehungen bei 30 Fällen; zusammen einige Sekunden (einmal gemessen, maschinen- und lastabhängig, kein Test).

COLOR_O = "#1f2733"
COLOR_NB = "#2a7de1"
COLOR_NBP = "#7fb2f0"
COLOR_IC = "#c77700"
COLOR_OV = "#8a96a8"
COLOR_NB0 = "#5b6b80"
RANKER_COLORS = {"O": COLOR_O, "NB": COLOR_NB, "NS3": "#34a89c", "NS10": "#9fd9d2", "NB0": COLOR_NB0, "NBp": COLOR_NBP, "IC": COLOR_IC, "OV": COLOR_OV}

# --- Presets (je eine Änderung gegenüber der Voreinstellung; die Zellen stehen in data/phn_results.json) ---------------------------------------
PRESETS = {
    "Wissensbasis bekannt": {**DEFAULT},
    "Seltene Krankheit (3 Fälle)": {**DEFAULT, "kb": 3},
    "Nur ein Fall": {**DEFAULT, "kb": 1},
    "Nicht alles geprüft": {**DEFAULT, "a": 0.2},
    "Viele Lücken": {**DEFAULT, "q": 0.6},
    "Falsch dokumentiert": {**DEFAULT, "fp": 0.05},
}
PRESET_HELP = {
    "Wissensbasis bekannt": "Die wahren Häufigkeiten sind bekannt, 200 Krankheiten, knapp ein Drittel (30 Prozent) der vorhandenen Merkmale wird nicht aufgeschrieben. Naive Bayes ist hier das Orakel.",
    "Seltene Krankheit (3 Fälle)": "Die Häufigkeiten werden aus nur 3 Fällen je Krankheit geschätzt, wie bei einer seltenen Krankheit mit wenigen Berichten.",
    "Nur ein Fall": "Ein einziger Fall je Krankheit als Wissensbasis.",
    "Nicht alles geprüft": "Nur ein Fünftel der Merkmale wurde überhaupt abgefragt; was fehlt, ist nicht dasselbe wie abwesend.",
    "Viele Lücken": "Mehr als die Hälfte der vorhandenen Merkmale wird nicht aufgeschrieben (Dokumentationsverlust 0,6).",
    "Falsch dokumentiert": "Zu fünf Prozent der nicht vorhandenen Merkmale steht ein falscher Eintrag in der Akte; die Wissensbasis kennt diese Fehler nicht.",
}
