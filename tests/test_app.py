"""AppTest-Rauchtests (DEMO-PLAYBOOK §1d Eckfall-Matrix): Standard, jedes Preset, jede Stufe jedes Reglers am Minimum und Maximum, ND 1000 mit 1 Fall, Kombination aller Extremwerte."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import phn_constants as C
import phn_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
STUFEN = [k for k, s in P.SETTING_SPECS.items() if s.options is not None]


def _fresh() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_laedt_ohne_ausnahme():
    at = _fresh()
    assert len(at.metric) >= 4


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_durch(name):
    at = _fresh()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == preset[key]


@pytest.mark.parametrize("state_key", STUFEN)
@pytest.mark.parametrize("kante", ["lo", "hi"])
def test_jede_stufe_am_rand_laeuft_durch(state_key, kante):
    at = _fresh()
    spec = P.SETTING_SPECS[state_key]
    at.sidebar.select_slider(key=state_key).set_value(spec.lo if kante == "lo" else spec.hi).run(timeout=TIMEOUT)
    assert not at.exception, (state_key, kante, [str(e) for e in at.exception])


def test_kombination_aller_minima_und_aller_maxima():
    for kante in ("lo", "hi"):
        at = _fresh()
        for state_key in STUFEN:
            spec = P.SETTING_SPECS[state_key]
            at.sidebar.select_slider(key=state_key).set_value(spec.lo if kante == "lo" else spec.hi)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (kante, [str(e) for e in at.exception])


def test_tausend_krankheiten_mit_einem_fall():
    at = _fresh()
    at.sidebar.select_slider(key="nd_select").set_value(1000)
    at.sidebar.select_slider(key="kb_select").set_value(1)
    at.run(timeout=TIMEOUT)
    assert not at.exception


def test_seed_und_patient_regler_am_rand():
    at = _fresh()
    at.sidebar.number_input(key="seed_input").set_value(0).run(timeout=TIMEOUT)
    assert not at.exception
    at = _fresh()
    at.sidebar.number_input(key="seed_input").set_value(C.SEED_MAX).run(timeout=TIMEOUT)
    assert not at.exception
    for wert in (1, 30):
        at = _fresh()
        at.slider(key="patient_idx").set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception


def test_beispielpatient_aufloesen_und_pravaelenz_tab():
    at = _fresh()
    at.toggle(key="patient_reveal").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Wahre Krankheit: K" in c.value for c in at.caption)
    for prev in (0.5, 0.1, 0.01, 0.001):
        at.select_slider(key="prev_select").set_value(prev).run(timeout=TIMEOUT)
        assert not at.exception
    at.radio(key="calib_ranker").set_value("O").run(timeout=TIMEOUT)
    assert not at.exception


def test_pdf_wird_erst_nach_dem_klick_erzeugt():
    at = _fresh()
    assert not [b for b in at.get("download_button")]
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_verdikt_und_kernfragen_stehen_auf_der_seite():
    at = _fresh()
    texte = [m.value for m in at.markdown]
    assert any("Kernfrage 1" in t for t in texte) and any("Kernfrage 2" in t for t in texte) and any("Kernfrage 3" in t for t in texte)
    assert any("keine medizinische Beratung" in t for t in texte)
