"""Reproduziert data/phn_results.json: Messreihe über alle Zellen (zwei Mengen von je fünf Welten), dazu der Abschnitt „seltene Klasse“.

Aufruf (im Demo-Ordner): python tools/sweep.py [--check]   (nicht in der CI; rund 1 bis 2 Minuten)
Mit --check wird nichts geschrieben, sondern die Datei gegen eine Neurechnung der Voreinstellung verglichen."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import phn_constants as C  # noqa: E402
import phn_live as L  # noqa: E402
import phn_results as RES  # noqa: E402
import phn_seltene as S  # noqa: E402


PAIRS = (("O", "NB"), ("NB", "NBp"), ("NB", "IC"), ("NB", "OV"), ("NBp", "IC"), ("NS3", "IC"), ("NS10", "IC"), ("NS3", "NB"), ("NS10", "NB"))


def cell_result(s: L.Settings) -> dict:
    m = L.measure(s, n_example=0)
    summ = L.summarize(m)
    nobs = L.by_nobs(m)
    out = {"summary": {r: {str(k): list(v) for k, v in summ[r].items()} for r in C.RANKERS},
           "nobs": {"share": nobs["share"], "recall5": nobs["recall5"]},
           "few": L.share_few(m),
           "paired": {f"{a}-{b}": list(L.paired_diff(m, a, b)) for a, b in PAIRS}}
    for r in ("O", "NB"):
        cal = L.calibration(m, r)
        out["calib_" + r] = {"mean_p": cal["mean_p"], "hit": cal["hit"], "bins": [list(b) for b in cal["bins"]]}
    return out


def main(check: bool = False):
    t0 = time.perf_counter()
    cells = {}
    for s in RES.sweep_settings():
        row = {"settings": {"nd": s.nd, "kb": s.kb, "q": s.q, "a": s.a, "fp": s.fp, "pr": s.pr}}
        for name, seed in C.SWEEP_SEEDS.items():
            row[name] = cell_result(L.Settings(nd=s.nd, kb=s.kb, q=s.q, a=s.a, fp=s.fp, pr=s.pr, seed=seed))
        cells[s.key()] = row
        print(f"{s.key():<40} NB R@5 A {row['A']['summary']['NB']['5'][0]:5.1f} B {row['B']['summary']['NB']['5'][0]:5.1f}", flush=True)
    seltene = {
        "k_auc": S.K_AUC, "true_auc": S.TRUE_AUC,
        "prevalence": [S.prevalence_row(31, p, nn, 40) for p, nn in ((0.5, 2000), (0.1, 5000), (0.01, 20000), (0.001, 200000))],
        "wilson": [S.wilson_coverage(32, n) for n in (10, 30, 100)],
        "spread": [S.auc_spread(33, n) for n in (10, 30, 100, 300)],
    }
    data = {"meta": {"worlds_per_set": C.N_WORLDS, "patients_per_world": C.N_PATIENTS, "sets": C.SWEEP_SEEDS, "cells": len(cells),
                     "rankers": list(C.RANKERS), "seconds": round(time.perf_counter() - t0, 1)},
            "cells": cells, "seltene": seltene}
    if check:
        old = json.loads(RES.DATA.read_text(encoding="utf-8"))
        a_old = old["cells"][L.Settings().key()]["A"]["summary"]["NB"]["5"][0]
        a_new = cells[L.Settings().key()]["A"]["summary"]["NB"]["5"][0]
        print("Prüfung Voreinstellung Menge A, Naive Bayes Recall@5:", a_old, a_new, "gleich" if abs(a_old - a_new) < 1e-9 else "ABWEICHUNG")
        return
    RES.DATA.parent.mkdir(exist_ok=True)
    RES.DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {RES.DATA} ({len(cells)} Zellen, {data['meta']['seconds']} s)")


if __name__ == "__main__":
    main("--check" in sys.argv)
