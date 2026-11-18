"""Auswertungs-PDF der aktuellen Messung (fpdf2, Kernschriften: keine Sonderzeichen außer Umlauten; Griechisch und Gedankenstrich ausgeschrieben)."""

import phn_constants as C
import phn_presets as P


def _ascii(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Gedankenstrich, Minuszeichen, Kreuz und typografische Anführungszeichen ersetzen."""
    table = {"–": "-", "—": "-", "−": "-", "×": "x", "„": '"', "“": '"', "”": '"', "’": "'", "≥": ">=", "≤": "<=", "±": "+/-"}
    return "".join(table.get(ch, ch) for ch in text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings, kennzahlen: dict, verdicts: tuple, n_worlds: int, n_patients: int) -> bytes:
    from fpdf import FPDF  # lazy: nur beim Klick

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _ascii("Phänotyp-Ranking: Auswertung einer Messung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _ascii("Lehrmodell mit erfundenen Daten, keine medizinische Beratung. Alle Zahlen gelten für diese Welt und diese Einstellung."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Einstellung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [("Krankheiten", P.label("nd_select", settings.nd)), ("Wissensbasis", P.label("kb_select", settings.kb)),
            ("Dokumentationsverlust", P.label("q_select", settings.q)), ("Abdeckung", P.label("a_select", settings.a)),
            ("falsche Einträge", P.label("fp_select", settings.fp)), ("Vorab", P.label("pr_select", settings.pr)),
            ("Zufalls-Seed", str(settings.seed)), ("Messung", f"{n_worlds} Welten x {n_patients} Patienten")]
    for k, v in rows:
        pdf.cell(60, 6, _ascii(k))
        pdf.cell(0, 6, _ascii(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _ascii("Recall in Prozent (Mittel ± Streuung über die Welten)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    for w, h in ((70, "Ranker"), (35, "Recall@1"), (35, "Recall@5"), (35, "Recall@10")):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    s = kennzahlen["summary"]
    for r in C.RANKERS:
        pdf.cell(70, 6, _ascii(C.RANKER_NAMES[r]), border=1)
        for k in (1, 5, 10):
            pdf.cell(35, 6, f"{s[r][k][0]:.1f} +/- {s[r][k][1]:.1f}".replace(".", ","), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    return bytes(pdf.output())
