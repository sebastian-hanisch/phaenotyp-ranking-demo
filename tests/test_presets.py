"""Presets: Vollständigkeit, gültige Stufen, Standard-Preset = Standardeinstellungen, jede Zelle der Presets ist gemessen; Permalink rastet auf Stufen ein."""

import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

import phn_constants as C
import phn_live as L
import phn_presets as P
import phn_results as RES

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_jedes_preset_hat_hilfe_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_auf_den_stufen_der_regler():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.options is not None:
                assert preset[key] in spec.options, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi


def test_erstes_preset_ist_die_voreinstellung():
    assert C.PRESETS["Wissensbasis bekannt"] == C.DEFAULT


def test_setting_specs_decken_jeden_preset_schluessel_ab():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for state_key, spec in P.SETTING_SPECS.items():
        assert spec.default in (spec.options or (spec.default,))


def test_jedes_preset_unterscheidet_sich_in_hoechstens_einem_regler_von_der_voreinstellung():
    for name, preset in C.PRESETS.items():
        diff = [k for k in C.DEFAULT if preset[k] != C.DEFAULT[k]]
        assert len(diff) <= 1, (name, diff)


def test_jede_preset_zelle_ist_in_der_messreihe_gemessen():
    for name, preset in C.PRESETS.items():
        s = RES.Settings(**{k: v for k, v in preset.items() if k != "seed"})
        assert RES.cell(s) is not None, name


def test_messreihe_deckt_alle_regler_stufen_ab():
    keys = {s.key() for s in RES.sweep_settings()}
    assert len(keys) == RES.load()["meta"]["cells"] == len(RES.sweep_settings())
    for nd in C.ND_OPTIONS:
        assert L.Settings(nd=nd).key() in keys
    for q in C.Q_OPTIONS:
        assert L.Settings(q=q).key() in keys
    for a in C.A_OPTIONS:
        assert L.Settings(a=a).key() in keys
    for fp in C.FP_OPTIONS:
        assert L.Settings(fp=fp).key() in keys
    for kb in C.KB_OPTIONS[1:]:
        assert L.Settings(kb=kb).key() in keys


def test_beschriftungen_gibt_es_fuer_jede_stufe():
    for state_key, spec in P.SETTING_SPECS.items():
        for o in spec.options or ():
            assert o in P.LABELS[state_key], (state_key, o)


def test_snap_rastet_auf_die_naechste_stufe_ein():
    spec = P.SETTING_SPECS["q_select"]
    assert P._snap(spec, 0.35) == 0.3 and P._snap(spec, 0.5) == 0.6 and P._snap(spec, -1) == 0.0


def test_permalink_rastet_ein_und_begrenzt():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["q"] = "0.35"
    at.query_params["kb"] = "7"
    at.query_params["seed"] = "99999999"
    at.run()
    assert not at.exception
    assert at.session_state["q_select"] == 0.3
    assert at.session_state["kb_select"] in (3, 10)       # 7 liegt zwischen 3 und 10: näher an 10 (Abstand 3 gegen 4)
    assert at.session_state["seed_input"] == C.SEED_MAX
