# Phänotyp-Ranking – welche seltene Krankheit passt? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-phaenotyp-ranking-demo.streamlit.app/)** -->

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Entscheidungsanalyse und Machine Learning). Aus wenigen **dokumentierten Merkmalen** (Symptomen) wird eine Rangliste möglicher **seltener Krankheiten** erstellt; die Demo vergleicht sieben Ranker (fünf davon in den Abbildungen, alle in den Tabellen) mit dem **Orakel** und zeigt, woran die Güte der Rangliste hängt: an der **Wissensbasis** (wie viele Fälle je Krankheit bekannt sind), an **Dokumentationslücken** (nicht aufgeschrieben, nicht geprüft, falsch aufgeschrieben) und an der Zahl der Merkmale je Patient.

**Lehrmodell mit erfundenen Daten, keine medizinische Beratung und keine Diagnoseempfehlung.** Welt, Krankheiten, Merkmale und alle Zahlen sind erzeugt; es gibt keine echten Patientendaten und keine echte Wissensbasis. Die Aussagen gelten für diese Welt, bei diesen Einstellungen und für die gemessene Regel jedes Rankers.

**Bezug zu OR:** ein Ranker ist eine Entscheidungshilfe unter Unsicherheit; die Frage, wie viel Wissen sie braucht und wann ein fehlender Eintrag etwas bedeutet, ist dieselbe wie bei Prognosen als Eingabe einer Planung (Wert der Information, Kalibrierung). Die Demo zeigt den Lernteil für sich.

## Warum dieses Problem

Bei seltenen Krankheiten sind Berichte dünn: je Krankheit gibt es wenige Fälle, aus denen sich die Häufigkeit der Merkmale schätzen lässt. **Kernfrage 1:** Wie viel Wissen braucht eine Rangliste, und ab wann liegt eine einfache Regel vor dem besseren Modell? **Kernfrage 2:** Ist ein nicht dokumentiertes Merkmal abwesend? **Kernfrage 3:** Wie wertet man einen Ranker bei wenigen Positiven aus (AUC gegen Average Precision, Wilson-Intervall, Streuung)?

## Befunde und Korrekturen gegenüber der früheren Messung

Eine frühere Messung vor dem Bau (NumPy-Zufall, fünf Welten × 1500 Patienten; sie ist nicht Teil dieses Repos, ihre Zahlen in der Spalte „alt“ sind hier nicht nachprüfbar) ging dem Bau voraus. Im Bau wurde mit dem **SplitMix64** der Demo in **zwei Mengen von je fünf Welten** (Zufalls-Seed 1000 und 2000) neu gemessen; der Zufallsstrom ist auf jeder Plattform derselbe, Gleitkomma-Rundung kann die Recall-Werte aber um unter 0,6 Punkte verschieben (siehe unter der Befundtabelle). Die Kernbefunde bestehen der Richtung nach, die Kreuzung nur mit der Laplace-Glättung (Korrektur 1) und „nicht geprüft ≠ abwesend“ erst bei Abdeckung 0,2 (Korrektur 2); die Recall-Zahlen verschieben sich um höchstens 3,2 Punkte (Häufigkeit × Informationsgehalt bei 30 Fällen), die Zahlen des Abschnitts „seltene Klasse“ stärker (Korrektur 7).

| Kernbeleg (Recall@5 in %, wenn nicht anders) | Frühere Messung (alt) | Neu gemessen (zehn Welten) |
|---|---|---|
| Voreinstellung: Naive Bayes = Orakel | 76,2 | 76,5 |
| Voreinstellung: nur dokumentierte / Häufigkeit × Informationsgehalt / Übereinstimmungen | 68,5 / 63,0 / 59,8 | 69,9 / 64,3 / 61,4 |
| Vorab nach Häufigkeit: Recall@1 Orakel / Naive Bayes ohne Vorab | 67,5 / 54,9 | 68,4 / 57,1 |
| Dokumentationsverlust 0 / 0,3 / 0,6: Orakel | 91,6 / 76,6 / 50,8 | 92,6 / 76,5 / 50,5 |
| 50 / 200 / 1000 Krankheiten: Orakel Recall@1 | 66,8 / 54,0 / 38,1 | 67,9 / 54,9 / 40,3 |
| 1 / 3 / 10 / 30 Fälle: Naive Bayes (Laplace) | 19,0 / 42,9 / 62,8 / 69,1 | 21,4 / 44,5 / 65,3 / 71,6 |
| 1 / 3 / 10 / 30 Fälle: Häufigkeit × Informationsgehalt | 31,4 / 46,8 / 56,5 / 59,5 | 33,3 / 49,4 / 59,6 / 62,7 |
| 1 / 3 / 10 / 30 Fälle: Zahl Übereinstimmungen | 2,5 / 2,5 / 38,5 / 49,2 | 2,5 / 2,5 / 40,8 / 51,4 |
| Abdeckung 0,5: Orakel / Naive Bayes / nur dokumentierte | 56,8 / 51,9 / 51,5 | 56,9 / 51,5 / 51,7 |
| Abdeckung 0,2: Orakel / Naive Bayes / nur dokumentierte | 42,4 / 34,3 / 38,5 | 41,8 / 34,3 / 38,6 |
| Kalibrierung, bekannte Wissensbasis: genannt / Treffer | 53,8 / 53,5 | 55,5 / 55,0 (Menge A) |
| Kalibrierung, 3 Fälle: genannt / Treffer | 18,1 / 24,6 | 18,9 / 25,4 (Menge A) |
| Average Precision bei Anteil Positiver 50 / 10 / 1 / 0,1 % | 0,951 / 0,766 / 0,409 / 0,138 | 0,925 / 0,590 / 0,125 / 0,014 |
| Streuung der AUC bei 10 / 30 / 100 / 300 Positiven | 0,030 / 0,019 / 0,010 / 0,006 | 0,016 / 0,009 / 0,005 / 0,003 |

**Korrekturen und Abweichungen gegenüber dem ersten Entwurf:**

1. **Die Kreuzung bleibt, hängt aber an der Laplace-Konstante:** Mit Laplace-Glättung liegt „Häufigkeit × Informationsgehalt“ bei 1 und 3 Fällen vor Naive Bayes, ab 10 Fällen Naive Bayes vor ihr; neu gemessen auch mit gepaarten Standardfehlern (Befund 6). In der früheren Messung stand bei n = 3 ein Abstand von 3,9 Punkten innerhalb der Streuung; gepaart über die Welten sind es 4,6 und 5,2 Punkte bei Standardfehlern von 1,3 und 0,8. Mit Schrumpfung 3 oder 10 gibt es die Kreuzung nicht (Befund 6). Mit fünf Welten je Menge ist „mehr als zwei Standardfehler“ ein Richtungskriterium, kein 95-Prozent-Test: bei vier Freiheitsgraden entspricht ein t von 2 einem zweiseitigen p von etwa 0,12.
2. **„Nicht geprüft ≠ abwesend“ bleibt, mit zwei Einschränkungen:** Bei Abdeckung 0,5 ist in diesen Welten kein Unterschied zwischen Naive Bayes und „nur dokumentierte Merkmale“ auflösbar (unter zwei Standardfehlern; das ist kein Beweis für Gleichheit), erst bei Abdeckung 0,2 liegt „nur dokumentierte Merkmale“ vorn. Der Vergleich mischt zwei Unterschiede: „nur dokumentierte Merkmale“ lässt auch die Abwesenheit geprüfter Merkmale weg; die Deutung „nicht dokumentiert ≠ abwesend“ ist plausibel, aber nicht isoliert gemessen.
3. **Bei bekannter Wissensbasis, voller Abdeckung und ohne falsche Einträge ist Naive Bayes per Konstruktion gleich dem Orakel** (in jeder der zehn Welten bestätigt); bei Abdeckung 0,2 oder falschen Einträgen ist es das nicht (Befunde 7 und 8). Die frühere Messung schätzte die „bekannte“ Wissensbasis aus 10.000 Fällen; hier sind es die wahren Häufigkeiten.
4. **Die Glättung ist kein Regler.** Ursprünglich war ein Regler „Glättung“ vorgesehen, der bei bekannter Wissensbasis wirkungslos wäre (toter Regler, DEMO-PLAYBOOK §3). Stattdessen laufen drei Naive-Bayes-Varianten nebeneinander (Laplace, Schrumpfung 3, Schrumpfung 10; dieselben Fälle). Die Pseudozählungen sind **gesetzt, nicht auf getrennten Welten abgestimmt**.
5. **Die Verdikte laufen live,** nicht aus der Messreihe: gepaarte Differenz über die Welten der aktuellen Messung gegen zwei Standardfehler; die Schwellen „reicht“ (Naive Bayes mindestens 90 Prozent des Orakels) und „zu dünn“ (unter 50 Prozent) sind gesetzte Faustregeln, keine Messung. Die Messreihe belegt die Aussagen der Kernfragen (Abbildungen, Tabellen).
6. **Neue Stufe „falsch dokumentiert“** (gegenüber der früheren Messung neu): Naive Bayes verliert gegen das Orakel bei 2 Prozent falschen Einträgen 0,5 / 0,7 Punkte Recall@5, bei 5 Prozent 1,8 / 1,5 Punkte (Menge A / B, gepaart, jeweils über zwei Standardfehler; Befund 8).
7. **Seltene Klasse mit anderen Scores:** Negative gleichverteilt, Positive mit Potenz 1/19 (wahre AUC genau 0,95, ohne Normalverteilung, ohne Logarithmus). Dadurch weichen die Average-Precision- und Streuungswerte von der früheren Messung (Normalverteilung) ab; **bei gleicher AUC hängt die Average Precision an der Form der Verteilung**, nicht nur an der Prävalenz.
8. **Rechenzeit der Live-Messung** höher als in der früheren Messung: fünf Welten × 1500 Patienten brauchen je nach Rechner und Last bei 200 Krankheiten etwa 0,6 bis 1,5 s, bei 1000 Krankheiten etwa 2 bis 4 s (Wissensbasis bekannt; einmal gemessen, kein Test), bei Schätzung aus 30 Fällen mehrere Sekunden mehr; die frühere Messung maß mit NumPy-Zufall und sechs Rankern 0,24 s und 1,0 s, hier sind es acht Ranker.

## Modell

**Welt.** 100 Merkmale mit Zipf-verteilter Beliebtheit $w_s \propto \mathrm{Rang}^{-0{,}8}$: beliebte Merkmale kommen in vielen Krankheitsprofilen und im Hintergrund vor (wenig Information). Jede Krankheit hat 4 bis 12 Profilmerkmale (gleichverteilt, gezogen ohne Zurücklegen mit Wahrscheinlichkeit proportional zur Beliebtheit) mit Häufigkeit 0,95 / 0,6 / 0,3 / 0,1 (Anteile 15 / 25 / 35 / 25 Prozent). Hintergrundrate $b_s = 0{,}01 + 0{,}2\,(w_s / w_{\max})^2$.

**Beobachtung.** Ein vorhandenes Merkmal wird mit Wahrscheinlichkeit $1 - q$ aufgeschrieben ($q$ = 0 / 0,3 / 0,6): $g_{ds} = 1 - (1 - f_{ds}(1 - q))(1 - b_s)$; Merkmale unabhängig gegeben die Krankheit. Abdeckung $a$: nur der Anteil $a$ der Merkmale wird geprüft, Nicht-Geprüftes steht als „nicht dokumentiert“. Falsche Einträge: ein nicht vorhandenes Merkmal steht mit Wahrscheinlichkeit 0 / 0,02 / 0,05 in der Akte (die Dokumentationswahrscheinlichkeit wird $1 - (1 - g_{ds})(1 - \text{Rate})$); die Wissensbasis kennt sie nicht. Patienten ohne jedes dokumentierte Merkmal werden nicht gerankt: in den zehn Welten bleiben 99,2 / 97,8 / 93,9 Prozent der Patienten bei $q$ = 0 / 0,3 / 0,6 und 97,8 / 84,9 / 53,4 Prozent bei Abdeckung 1,0 / 0,5 / 0,2; **die Recall-Werte der Befunde gelten für diese gerankten Patienten** (zählten die übrigen als Fehlschlag, läge der Recall bei Abdeckung 0,2 deutlich niedriger).

**Vorab.** gleichverteilt oder Zipf-Verteilung mit Exponent 1 („nach Häufigkeit“).

**Wissensbasis.** wahre Häufigkeiten („bekannt“) oder geschätzt aus 1 / 3 / 10 / 30 gezogenen Fällen je Krankheit: Laplace $\hat g = (c + 0{,}5)/(n + 1)$ oder Schrumpfung zum Merkmalsmittel $\hat g = (c + a\,\bar m)/(n + a)$ mit Pseudozählung $a$ = 3 oder 10.

**Orakel.** Der exakte Posterior des Erzeugers mit den wahren Häufigkeiten, dem wahren Vorab, der Maske der geprüften Merkmale und den falschen Einträgen: $P(d \mid x, m) \propto \pi_d \prod_{s: m_s = 1} g_{ds}^{x_s}(1 - g_{ds})^{1 - x_s}$. **Das Orakel gehört zum Erzeuger und ist nur in diesem Modell eine Obergrenze**; die Naive-Bayes-Struktur (Merkmale unabhängig gegeben die Krankheit) steckt im Erzeuger, deshalb ist es im Erwartungswert das beste Verfahren dieser Welt. In einer Stichprobe von 1500 Patienten liegt es nicht strikt über jedem anderen Ranker: In allen 896 Vergleichen (14 Zellen × 2 Mengen × 8 Ranker × 4 Kennzahlen) übertrifft ein Ranker das Orakel um höchstens 0,03 Punkte, Rundungsreste bei Gleichständen (am größten bei „Naive Bayes ohne Vorab“, das bei gleichverteiltem Vorab mathematisch dasselbe rechnet).

## Methodik

Acht Ranker: Orakel; Naive Bayes mit Laplace-Glättung; Naive Bayes mit Schrumpfung 3 und 10; Naive Bayes ohne Vorab; „nur dokumentierte Merkmale“ (Abwesenheit zählt nicht); „Häufigkeit × Informationsgehalt“ ($\sum_s x_s \hat g_{ds} \mathrm{IC}_s$ mit $\mathrm{IC}_s = -\ln \bar g_s$, anschließend um das Minimum über alle Merkmale verschoben, sodass der kleinste Wert 0,001 ist; Phenomizer-artig, nicht der Originalalgorithmus); „Zahl Übereinstimmungen“ (dokumentierte Merkmale mit geschätzter Häufigkeit über 0,05).

**Kennzahlen.** Recall@1, @5, @10: Anteil der Patienten, bei denen die wahre Krankheit unter den ersten $k$ steht; Gleichstände werden als Wahrscheinlichkeit gezählt (zufällige Auflösung, $\min(1, \max(0, (k - r + 1)/(t + 1)))$ mit dem kleinsten Rang $r$ und $t$ Gleichständen), nicht als Treffer. Dazu der **reziproke mittlere Rang** $1/(r + t/2)$ (kleinster Rang $r$ plus Hälfte der $t$ Gleichstände); das ist nicht der Erwartungswert von 1/Rang bei zufälliger Auflösung, der bei Gleichständen größer ist (bei drei Gleichständen auf Rang 1 gilt $1/2$ gegen $(1 + 1/2 + 1/3)/3 \approx 0{,}61$). **Einheit:** Patienten (Recall) beziehungsweise Welten (Streuung).

**Messung.** Je Einstellung 5 Welten × 1500 Patienten (SplitMix64, `phn_stream.py` ist die vektorisierte Fassung der Skalarklasse in `phn_rng.py`, bitgleich geprüft). Mittel ± Streuung (n − 1) über die Welten. **Gepaarte Vergleiche:** Ob ein Ranker vor einem anderen liegt, entscheidet die Differenz je Welt (Mittel ± Standardfehler = Streuung der Differenzen / √5); Unterschiede unter zwei Standardfehlern werden nicht gedeutet (ein Richtungskriterium bei fünf Welten, kein 95-Prozent-Test, Korrektur 1). **Vorgerechnet** (`data/phn_results.json`, `tools/sweep.py`): 14 Zellen (Voreinstellung und je eine Abweichung: Vorab, q ×2, 50 und 1000 Krankheiten, 1 / 3 / 10 / 30 Fälle, Abdeckung ×2, falsche Einträge ×2), je zwei Mengen von fünf Welten (Zufalls-Seed 1000 und 2000).

**Seltene Klasse** (`phn_seltene.py`): Scores Negative $U \sim [0, 1)$, Positive $U^{1/19}$ (wahre AUC $= 19/20 = 0{,}95$ exakt); Average Precision, PPV bei Sensitivität 90 %, Wilson- gegen Wald-Intervall, Streuung der AUC bei wenigen Positiven.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: ND = 200 Krankheiten, 100 Merkmale, Dokumentationsverlust q = 0,3, Vorab gleichverteilt, alle Merkmale geprüft, keine falschen Einträge, wenn nicht anders genannt; Recall@5 in Prozent der Patienten, **Mittel der zwei Mengen zu je 5 Welten × 1500 Patienten (zehn Welten)**; „gepaart“ = Mittel der Differenzen je Welt in Punkten, Menge A / Menge B. Jede Zeile nennt den Test in `tests/test_claims.py`.

| # | Frage | Befund | Test |
|---|---|---|---|
| 1 | Wissensbasis bekannt: wie gut ist die Rangliste? | Recall@5: Orakel = Naive Bayes 76,5; nur dokumentierte Merkmale 69,9; Häufigkeit × Informationsgehalt 64,3; Zahl Übereinstimmungen 61,4. Recall@1: 54,9 / 48,4 / 41,3 / 38,8. Recall@10: 83,8 / 77,9 / 74,2 / 70,8. Reziproker mittlerer Rang: 64,9 / 58,0 / 51,8 / 44,9. Naive Bayes ist in jeder der zehn Welten gleich dem Orakel (Unterschied 0; das ist Konstruktion, kein Messbefund, siehe Ehrliche Grenzen). Gepaart Naive Bayes minus Häufigkeit × Informationsgehalt: +12,2 / +12,1 (Standardfehler 0,4 / 0,6); minus nur dokumentierte: +6,4 / +6,7; minus Übereinstimmungen: +15,4 / +14,9. | `test_befund_1_*` |
| 2 | Wie viel bringt das Vorab? | Vorab nach Häufigkeit (Zipf): Recall@1 Orakel 68,4 gegen Naive Bayes ohne Vorab 57,1; Recall@5 86,1 gegen 77,6. Bei gleichverteiltem Vorab (Voreinstellung) bringt es nichts: Naive Bayes mit und ohne Vorab gleich (76,5); das folgt aus der Rechnung (ein konstanter Vorab-Term ändert die Rangliste nicht) und ist kein Messbefund. Gemessen ist nur die Wirkung des Zipf-Vorab mit Exponent 1 bei 200 Krankheiten. | `test_befund_2_*` |
| 3 | Dokumentationsverlust? | Orakel Recall@5 bei q = 0 / 0,3 / 0,6: 92,6 / 76,5 / 50,5. Anteil der Patienten mit höchstens zwei dokumentierten Merkmalen: 13,5 / 25,5 / 45,0 Prozent. Beides bezieht sich auf die gerankten Patienten (99,2 / 97,8 / 93,9 Prozent aller Patienten); je größer der Verlust, desto mehr Patienten ohne Eintrag fehlen, was den Recall leicht beschönigt. | `test_befund_3_*` |
| 4 | Zahl der Krankheiten? | Orakel Recall@5 bei 50 / 200 / 1000 Krankheiten: 88,0 / 76,5 / 60,0; Recall@1: 67,9 / 54,9 / 40,3. | `test_befund_4_*` |
| 5 | Zahl dokumentierter Merkmale? | Anteil der Patienten bei 1–2 / 3–4 / 5–7 / ab 8 Merkmalen: 25,5 / 42,3 / 29,0 / 3,2 Prozent. Orakel Recall@5 dort: 54,6 / 77,8 / 91,5 / 97,8; nur dokumentierte 43,5 / 71,1 / 88,4 / 97,4; Häufigkeit × Informationsgehalt 40,1 / 63,8 / 82,9 / 94,7. | `test_befund_5_*` |
| 6 | **Wissensbasis aus wenigen Fällen** (1 / 3 / 10 / 30 Fälle je Krankheit; Orakel immer 76,5) | Naive Bayes (Laplace) 21,4 / 44,5 / 65,3 / 71,6; Schrumpfung 3: 36,4 / 54,9 / 65,1 / 67,5; Schrumpfung 10: 35,3 / 54,1 / 66,5 / 70,5; nur dokumentierte 28,8 / 49,0 / 61,3 / 65,7; Häufigkeit × Informationsgehalt 33,3 / 49,4 / 59,6 / 62,7; Zahl Übereinstimmungen 2,5 / 2,5 / 40,8 / 51,4 (die 2,5 bei 1 und 3 Fällen sind ein Artefakt der Laplace-Schätzung und kein Ergebnis über die Regel: dort ist die kleinste geschätzte Häufigkeit 0,25 beziehungsweise 0,125, also über der Schwelle 0,05; jedes dokumentierte Merkmal zählt bei jeder Krankheit, alle Krankheiten bekommen dieselbe Punktzahl, und Recall@5 ist 5 von 200 Krankheiten, also 2,5 Prozent; ab 10 Fällen liegt die kleinste Häufigkeit mit 0,045 unter der Schwelle und die Regel trennt). **Gepaart Naive Bayes (Laplace) minus Häufigkeit × Informationsgehalt:** −11,4 / −12,6 bei 1 Fall, −4,6 / −5,2 bei 3 Fällen (jeweils mehr als zwei Standardfehler zugunsten der einfachen Regel), +5,9 / +5,4 bei 10, +9,0 / +8,9 bei 30 Fällen (zugunsten von Naive Bayes). Mit Schrumpfung 3 liegt Naive Bayes dagegen bei 1 und 3 Fällen vor der einfachen Regel (+3,0 / +3,1 und +5,8 / +5,2), mit Schrumpfung 10 ebenfalls (+1,9 / +2,0 und +4,8 / +4,6). Schrumpfung 3 minus Laplace: +14,4 / +15,6 (1 Fall), +10,4 / +10,4 (3), +0,1 / −0,4 (10), −4,3 / −4,0 (30); Schrumpfung 10 minus Laplace bei 30 Fällen −1,1 / −1,1. **Die Schätzung ist ein Regler, keine Einstellung gewinnt überall.** | `test_befund_6_*` |
| 7 | **Nicht geprüft = abwesend?** (Abdeckung 1,0 / 0,5 / 0,2) | Recall@5 Orakel 76,5 / 56,9 / 41,8; Naive Bayes 76,5 / 51,5 / 34,3; nur dokumentierte Merkmale 69,9 / 51,7 / 38,6. Anteil der Patienten mit höchstens zwei Merkmalen 25,5 / 65,9 / 92,9 Prozent. Gerankt werden 97,8 / 84,9 / 53,4 Prozent der Patienten (die übrigen haben kein dokumentiertes Merkmal); die Recall-Werte gelten für die gerankten. Gepaart Naive Bayes minus nur dokumentierte: bei 1,0 +6,4 / +6,7; bei 0,5 −0,4 / +0,1 (Standardfehler 0,4 / 0,5: kein auflösbarer Unterschied, kein Beweis für Gleichheit); bei 0,2 −4,5 / −4,2 (Standardfehler 1,1 / 0,5: „nur dokumentierte“ liegt vorn). | `test_befund_7_*` |
| 8 | Falsche Einträge (0 / 0,02 / 0,05) | Recall@5 Orakel 76,5 / 67,4 / 58,4; Naive Bayes 76,5 / 66,8 / 56,7; nur dokumentierte 69,9 / 59,7 / 49,8. Gepaart Orakel minus Naive Bayes: +0,5 / +0,7 (bei 0,02), +1,8 / +1,5 (bei 0,05); Naive Bayes minus nur dokumentierte: +7,4 / +6,6 und +6,9 / +7,0. | `test_befund_8_*` |
| 9 | Kalibrierung der Spitzen-Wahrscheinlichkeit (Naive Bayes, Menge A) | Bekannte Wissensbasis: mittlere genannte Wahrscheinlichkeit 55,5 Prozent, Trefferquote 55,0; in allen vier Fächern (bis unter 0,5 / 0,9 / 0,99 / bis 1) unterscheiden sich genannte Wahrscheinlichkeit und Trefferquote um weniger als 0,03. Aus 3 Fällen (Laplace): 18,9 / 25,4; im Fach 0,5 bis unter 0,9 (415 Patienten) genannt 0,652, Treffer 0,875: **unterkonfident**. Die Richtung hängt an der Schätzung (zehn Welten, 3 Fälle, genannt / Treffer in Prozent): Laplace 18,6 / 25,6, Schrumpfung 3 35,3 / 32,7 (überkonfident), Schrumpfung 10 16,0 / 32,5. Die bekannte Wissensbasis ist nur ohne Lücken kalibriert (das folgt aus der Theorie, der exakte Posterior ist kalibriert, Konstruktion): Naive Bayes mit wahren Häufigkeiten nennt bei Abdeckung 0,2 23,4 bei 14,3 Prozent Treffern, bei 5 Prozent falschen Einträgen 58,7 bei 33,4, bei Dokumentationsverlust 0,6 28,4 bei 28,0; das Orakel nennt 17,6 bei 17,4 (Abdeckung 0,2) und 35,1 bei 34,8 (falsche Einträge). | `test_befund_9_*` |
| 10 | Stabilität | Die beiden Mengen von Welten unterscheiden sich im Recall@5 um höchstens 3,02 Punkte (Häufigkeit × Informationsgehalt in der Zelle „Vorab nach Häufigkeit“; geprüft über 14 Zellen × 8 Ranker); die Streuung über die Welten beträgt höchstens 5,7 Punkte (über alle Zellen, Ranker und beide Mengen). | `test_stabilitaet_*` |
| 11 | AUC gegen Average Precision (Scores mit wahrer AUC 0,95; Anteil Positiver 50 / 10 / 1 / 0,1 Prozent; 2000 / 556 / 202 / 200 Positive, 2000 / 5000 / 20.000 / 200.000 Negative; 40 Ziehungen) | AUC 0,951 / 0,951 / 0,950 / 0,950; Average Precision 0,925 / 0,590 / 0,125 / 0,014; PPV bei Sensitivität 90 Prozent 0,889 / 0,472 / 0,074 / 0,008. | `test_befund_11_*` |
| 12 | Wilson gegen Wald (Sensitivität 0,9, 95-Prozent-Intervall, 20.000 Wiederholungen, 10 / 30 / 100 Positive) | Überdeckung Wald 0,646 / 0,810 / 0,932; Wilson 0,930 / 0,976 / 0,936; mittlere Breite Wald 0,279 / 0,201 / 0,116, Wilson 0,369 / 0,215 / 0,118. | `test_befund_12_*` |
| 13 | Streuung der AUC bei wenigen Positiven (Anteil 1 Prozent, 400 Datensätze, 10 / 30 / 100 / 300 Positive) | Standardabweichung 0,0156 / 0,0095 / 0,0048 / 0,0028; Abweichung über 0,03 von der wahren AUC in 400 Datensätzen in 5,5 / 0,0 / 0,0 / 0,0 Prozent beobachtet (kein Beleg für „nie“: bei 30 Positiven liegt 0,03 bei 3,2 Standardabweichungen, nach Normalnäherung wären etwa 0,6 von 400 Datensätzen zu erwarten). | `test_befund_13_*` |

Die Zahlen sind in der Datei und im Test auf eine Stelle gerundet; `test_neu_rechnen_*` rechnet die Voreinstellung und die Zelle „3 Fälle“ (Menge A) neu und vergleicht im Band von 0,6 Punkten (Linux rechnet an der letzten Binärstelle anders); der Zufallsstrom selbst ist bitgleich (`tests/test_stream.py`), die Recall-Werte können sich durch Gleitkomma-Rundung (BLAS, libm) um unter 0,6 Punkte unterscheiden. Was **nicht** im Test steht: die Rechenzeiten (Korrektur 8), die Zahlen der Spalte „alt“ (Messung vor dem Bau, nicht Teil dieses Repos) und die Zahl der Tests selbst.

## Ehrliche Grenzen

- **Das Orakel gehört zum Erzeuger.** Die Naive-Bayes-Struktur steckt in der Welt; dass Naive Bayes bei bekannter Wissensbasis, voller Abdeckung und ohne falsche Einträge das Orakel erreicht, ist Konstruktion, kein Befund (die Umsetzung prüft `tests/test_oracle_phaenotyp_ranking.py`). Ebenso folgen aus der Konstruktion das Vorzeichen „Orakel ≥ Naive Bayes“ in den Befunden 7 und 8 (Optimalität des Posteriors), die Kalibrierung bei bekannter Wissensbasis ohne Lücken (Befund 9) und die Gleichheit von Naive Bayes mit und ohne Vorab bei gleichverteiltem Vorab (Befund 2). **Nicht aus der Konstruktion ableitbar** sind die Größen der Abstände in den Befunden 6 bis 8 und die Kalibrierung bei geschätzter Wissensbasis.
- **Echte Merkmalstabellen** haben Abhängigkeiten und Hierarchien (Ontologie), Schweregrad, Alter und Verlauf; hier sind Merkmale unabhängig gegeben die Krankheit und nur an oder aus.
- **Größe:** 50 bis 1000 Krankheiten und 100 Merkmale, nicht Tausende; zwei Mengen von je fünf Welten. Die Streuung über die Welten und gepaarte Standardfehler stehen bei den Zahlen; Unterschiede unter zwei Standardfehlern werden nicht gedeutet.
- **Die Schätzungen sind gesetzt:** Laplace 0,5 je Seite, Schrumpfung 3 und 10 (Korrektur 4). Die Aussage „keine Schätzung gewinnt überall“ gilt für diese drei Einstellungen und die vier Fallzahlen.
- **„Häufigkeit × Informationsgehalt“** ist eine vereinfachte Fassung im Stil bekannter Ähnlichkeitsmaße, nicht ein bestimmtes Werkzeug; Aussagen über sie gelten für diese Regel.
- **Falsche Einträge** als gleichmäßige Rate je nicht vorhandenem Merkmal; echte Fehler häufen sich bei bestimmten Merkmalen.
- **Seltene Klasse:** Scores mit bekannter AUC, keine Ranker dieser Welt; die Streuung der AUC ist die eines Testsatzes mit festem Score, keine Kreuzvalidierung mit wiederholter Anpassung. Die Average Precision hängt bei gleicher AUC an der Form der Verteilung (Korrektur 7).
- **Stand der Technik** (agentische LLM-Systeme, Vergleichswerkzeuge für Phänotyp und Genom) wurde nur aus Zusammenfassungen gelesen und ist hier **nicht** belegt oder verglichen.
- **Bekannte Probleme:** nur an erfundenen Welten geprüft; die Schätzung aus 30 Fällen bei 1000 Krankheiten rechnet in der Live-Ansicht mehrere Sekunden (mit Wartehinweis, Ergebnis wird je Einstellung zwischengespeichert).

## Tests

197 Tests (`pytest tests/`, mit `-n 8` rund eine Minute): Zufallsstrom gegen die Skalarklasse (bitgleich) und Kopie der Vorlage; Welt (Form, Wertebereiche, Dokumentationsformel von Hand, Abdeckung, falsche Einträge, Wissensbasis); **Orakel** (Posterior schleifenweise in reinem Python gegen die vektorisierten Ranker auf der Mini-Welt mit 16 Mustern; exakter Erwartungswert des Recalls durch Aufzählung; AUC gegen paarweises Zählen; Average Precision nach Definition; Wilson gegen die Lösung der quadratischen Gleichung); Kennzahlen und Randfälle (Gleichstände, Rangsumme, leere Fächer); Live-Messung (Wiederholbarkeit, **Zweig-Test je Regler**: jede Stufe ändert mindestens einen Ranker, Naive Bayes = Orakel bei bekannter Wissensbasis, Worst-Case am Regler-Maximum ohne Zeit-Assertion); Verdikte (Schwellen an handgesetzten Werten, jede Zahl im Text) und **je Preset die erwartete Verdikt-Zeile**; Abnahmekriterien der Presets an der Messreihe, jedes an künstlichen Werten einzeln kippend; Zahlen dieser README (`test_claims.py`); AppTest (Standard, jedes Preset, jede Stufe jedes Reglers am Minimum und Maximum, Kombination aller Minima und aller Maxima, 1000 Krankheiten mit 1 Fall, PDF erst nach Klick); Layout, Footer, Laufzeit-Importe. Keine Mutationstests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Git- und Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `phn_constants.py` | Annahmen der Welt, Regler-Stufen, Ranker, Farben, Presets |
| `phn_rng.py` | SplitMix64 (unveränderte Kopie der Vorlage aus dem Website-Repo) |
| `phn_stream.py` | vektorisierte Fassung desselben Stroms, abgeleitete Zufalls-Seeds |
| `phn_world.py` | Erzeuger: Profile, Hintergrund, Patienten, Wissensbasis aus Fällen |
| `phn_rankers.py` | Orakel, Naive Bayes (drei Schätzungen), ohne Vorab, nur dokumentierte, Häufigkeit × Informationsgehalt, Übereinstimmungen |
| `phn_metrics.py` | Recall@k mit Gleichständen, reziproker Rang, Kalibrierung, AUC, Average Precision, Wilson |
| `phn_live.py` | Live-Messung, Auswertung über die Welten, gepaarte Differenz |
| `phn_evaluation.py` | Verdikte (reine Funktionen), Beispielpatient, Zahlenformat |
| `phn_seltene.py` | Auswertung bei seltener Klasse |
| `phn_results.py`, `data/phn_results.json` | Zugriff auf die vorgerechnete Messreihe (14 Zellen × 2 Mengen, Abschnitt „seltene Klasse“) |
| `phn_stories.py` | Abnahmekriterien der Presets |
| `phn_presets.py`, `phn_layout.py` | Stufen-Regler, Permalink (rastet auf Stufen ein), Schnellstart, Hervorhebung, fester Kopf |
| `phn_visualization.py`, `phn_ui_panel.py`, `phn_pdf_export.py` | Abbildungen (Achsen gesperrt), Darstellung, PDF (fpdf2, erst nach Klick) |
| `tools/sweep.py` | reproduziert `data/phn_results.json` (rund eine Minute, nicht in der CI) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit beziehungsweise plus pytest |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

Ontologie-Hierarchie der Merkmale (Eltern-Kind-Beziehungen), Abhängigkeiten zwischen Merkmalen, Schweregrad und Alter, Genvarianten und Bildmerkmale, echte Wissensbasen (HPO, Orphanet), ein LLM-Ranker (Schlüssel, Kosten, Reproduzierbarkeit und Datenschutz sprechen gegen eine Live-Demo; höchstens als Erklärseite nach Lesen der Quellen im Wortlaut), Abstimmung der Schrumpfung auf getrennten Welten, mehr als 1000 Krankheiten.

## Verwandte Demos mit demselben mathematischen Modell

- [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo): Kalibrierung und Schwelle aus dem Kostenverhältnis (Logit auf Container-Merkmalen); hier wird die genannte Wahrscheinlichkeit des Spitzenreiters gegen die Trefferquote gehalten, dort die Entscheidung aus einer Wahrscheinlichkeit.
- [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo): Bayes-Update und Wert eines Tests bei zwei Zuständen; hier sind es viele Krankheiten und ein Ranking.
- `diagnose-testfolge-demo` (Schwesterdemo): welcher Test als Nächster; dort ist das Vorab der Diagnosen eine Eingabe, die Demos sind nicht verbunden.
- [`cart-demo`](https://github.com/sebastian-hanisch/cart-demo): Klassifikation aus Merkmalen mit einem Entscheidungsbaum, anderes Modell.
- [`isolation-forest-demo`](https://github.com/sebastian-hanisch/isolation-forest-demo): Anomalie-Erkennung als Ein-Klassen-Sicht auf Seltenes; hier gibt es für jede Krankheit ein Profil.

Zahlen aus diesen Demos werden hier weder nachgerechnet noch verglichen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Python-Version der Streamlit Cloud ist in der App-Einstellung zu wählen (die CI läuft mit Python 3.12); die Demo nutzt nur Sprachmerkmale ab Python 3.10. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
