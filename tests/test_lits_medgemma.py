"""Tests for the LiTS17 data layer, the MedGemma second-reader tool and the fused calibration.

None of these need a GPU, PyTorch, nibabel or a Hugging Face download: the LiTS files are small
phantoms written in the Kaggle layout and the models are the stand-in plugins in tests/fixtures.

Run:  pytest -q
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radagent.calibration import Calibrator, fit_calibrator, nll, sigmoid  # noqa: E402
from radagent.config import load_config  # noqa: E402
from radagent.data.lits import (build_manifest, case_number, discover_pairs, locate,  # noqa: E402
                                notebook_compatible_split)
from radagent.io.nifti import _to_compact  # noqa: E402
from radagent.vlm.ct_prep import liver_z_range, select_slices, window_rgb  # noqa: E402
from radagent.vlm.guard import check_text, parse_final_answer, unverified_measurements  # noqa: E402
from radagent.vlm.medgemma import single_token_ids, yes_probability  # noqa: E402
from radagent.vlm.prompts import build_content  # noqa: E402
from tests.fixtures.phantoms import write_lits_like  # noqa: E402


# --------------------------------------------------------------------------- LiTS data layer
def test_notebook_compatible_split_matches_the_vitunetr_notebook():
    split = notebook_compatible_split(list(range(131)))
    counts = {k: sum(v == k for v in split.values()) for k in ("train", "val", "calib", "test")}
    assert counts == {"train": 68, "val": 12, "calib": 26, "test": 25}
    assert split == notebook_compatible_split(list(range(131)))       # deterministic


def test_kaggle_layout_discovery_manifest_and_locate(tmp_path):
    root = write_lits_like(tmp_path / "lits", n_cases=6, size=96, seed=3)
    pairs = discover_pairs([root])
    assert sorted(pairs) == list(range(6))
    for n, (ct, seg) in pairs.items():
        assert case_number(ct) == n == case_number(seg)
        assert Path(ct).name == f"volume-{n}.nii" and Path(seg).name == f"segmentation-{n}.nii"
    assert {Path(ct).parent.parent.name for ct, _ in pairs.values()} == {"part1", "part2"}

    cfg = load_config(None)
    cfg["data"].update({"lits_dirs": [str(root)], "split_strategy": "random",
                        "split_fractions": {"train": 0.34, "val": 0.16, "calib": 0.25, "test": 0.25}})
    man = build_manifest(cfg, allow_download=False)
    assert list(man["case_id"]) == [f"lits_{i:03d}" for i in range(6)]
    assert set(man["split"]) <= {"train", "val", "calib", "test"} and (man["split"] == "test").any()
    assert all(Path(p).exists() for p in man["ct_path"])

    # a run made elsewhere (e.g. on Colab) points at a path that does not exist here
    moved = "/content/kaggle/liver-tumor-segmentation/volume_pt1/" + Path(pairs[0][0]).name
    assert Path(locate(moved, cfg)).resolve() == Path(pairs[0][0]).resolve()
    assert locate("/nowhere/volume-999.nii", cfg) == "/nowhere/volume-999.nii"


def test_float_ct_is_stored_compactly():
    ints = [np.array([[-1024.0, 0.0], [12.0, 3071.0]], dtype=np.float32)]
    a = _to_compact(ints, (2, 2))
    assert a.dtype == np.int16 and a[1, 1] == 3071
    frac = [np.array([[0.5, 1.0]], dtype=np.float64)]
    assert _to_compact(frac, (1, 2)).dtype == np.float32


# --------------------------------------------------------------------------- MedGemma tool pieces
def test_ct_window_rgb_and_slice_selection():
    hu = np.array([[-2000, -135, 40, 215, 3000]], dtype=np.int16)
    rgb = window_rgb(hu, [[-1024, 1024], [-135, 215], [0, 80]])
    assert rgb.shape == (1, 5, 3) and rgb.dtype == np.uint8
    assert list(rgb[0, :, 1]) == [0, 0, 128, 255, 255]                # soft-tissue window
    assert list(rgb[0, :, 2]) == [0, 0, 128, 255, 255]                # 0..80 HU window
    zs = select_slices(10, 90, 16)
    assert len(zs) == 16 and zs[0] == 90 and zs[-1] == 10 and zs == sorted(zs, reverse=True)
    assert select_slices(5, 7, 16) == [7, 6, 5]
    assert liver_z_range({"measurements": {"liver_slice_range": [3, 40]}}) == (3, 40)
    assert liver_z_range({"measurements": {}, "segmentation": {"found_liver": False}}) is None


def test_prompt_layout_follows_the_ct_recipe():
    items = [{"type": "image", "image": f"img{i}"} for i in range(3)]
    c = build_content(3, "Q?", items)
    assert c[0]["type"] == "text" and "3 axial CT slices" in c[0]["text"]
    assert [x.get("text") for x in c[2:7:2]] == ["SLICE 1", "SLICE 2", "SLICE 3"]
    assert c[-1]["text"] == "Q?" and sum(x["type"] == "image" for x in c) == 3


def test_yes_probability_from_first_token_logits():
    logits = np.full(50, -5.0)
    logits[[3, 4]] = [2.0, 1.0]           # yes variants
    logits[[7]] = [0.5]                    # no
    p, mass = yes_probability(logits, [3, 4], [7])
    expect = (math.exp(2) + math.exp(1)) / (math.exp(2) + math.exp(1) + math.exp(0.5))
    assert math.isclose(p, expect, rel_tol=1e-9) and 0.5 < mass < 1.0

    class Tok:                             # 'yes'/'no' single tokens, ' yes' splits in two
        vocab = {"yes": 11, "Yes": 12, "no": 21, "No": 22, "YES": 13, "NO": 23}

        def encode(self, s, add_special_tokens=False):
            return [self.vocab[s]] if s in self.vocab else [99, 100]

    assert single_token_ids(Tok(), ["yes"]) == [11, 12, 13]
    assert single_token_ids(Tok(), ["no"]) == [21, 22, 23]


def test_guard_flags_sizes_the_measurement_tool_did_not_produce():
    meas = {"lesions": [{"max_axial_diameter_mm": 24.0, "volume_ml": 6.2}], "predicted_tumour_volume_ml": 6.2,
            "liver_volume_ml": 1500.0}
    assert unverified_measurements("lesion of 2.5 cm and 6 ml", meas) == []
    assert unverified_measurements("a 9 cm mass", meas) == ["9 cm"]
    assert parse_final_answer("blah. Final Answer: **Yes**") == 1 and parse_final_answer("nothing") is None
    g = check_text("No lesion seen. Final Answer: no", meas, p_yes=0.9)
    assert g["final_answer"] == 0 and any("disagrees" in n for n in g["notes"])


# --------------------------------------------------------------------------- fused calibration
def test_fused_calibrator_learns_from_the_second_reader():
    rng = np.random.default_rng(5)
    n = 120
    y = (rng.random(n) < 0.5).astype(int)
    p_seg = sigmoid(rng.normal(0, 1, n) + 0.6 * (2 * y - 1))         # weak segmenter
    p_vlm = sigmoid(rng.normal(0, 1, n) + 2.0 * (2 * y - 1))         # informative second reader
    cal, info = fit_calibrator(p_seg, y, "auto", p_vlm=p_vlm)
    assert info["selected"] in ("vlm", "fused") and info["n_with_vlm"] == n
    assert nll(cal(p_seg, p_vlm), y) < nll(cal.seg_only(p_seg), y)
    # no stored second read for a case -> the segmenter-only map is used for it
    assert math.isclose(cal(float(p_seg[0])), float(cal.seg_only(p_seg[0])))
    mixed = np.where(np.arange(n) % 2 == 0, p_vlm, np.nan)
    out = cal(p_seg, mixed)
    assert np.allclose(out[1::2], cal.seg_only(p_seg[1::2]))
    assert np.allclose(Calibrator.from_dict(cal.to_dict())(p_seg, p_vlm), cal(p_seg, p_vlm))
    cal2, info2 = fit_calibrator(p_seg, y, "auto")                      # without a VLM: old behaviour
    assert info2["selected"] in ("identity", "temperature", "platt") and not cal2.has_vlm


# --------------------------------------------------------------------------- whole pipeline
def _e2e_cfg(tmp_path):
    root = write_lits_like(tmp_path / "lits", n_cases=12, size=112, seed=11)   # both classes in calib
    cfg = load_config(None)
    cfg["run_name"], cfg["output_root"] = "e2e", str(tmp_path / "outputs")
    cfg["data"].update({"lits_dirs": [str(root)], "split_strategy": "random",
                        "split_fractions": {"train": 0.17, "val": 0.08, "calib": 0.42, "test": 0.33}})
    cfg["preprocess"]["img_size"] = 112
    cfg["segmenter"].update({"backend": "plugin", "plugin": "tests.fixtures.stubs:ThresholdSegmenter"})
    cfg["vlm"].update({"backend": "plugin", "plugin": "tests.fixtures.stubs:StubVLM", "n_slices": 6})
    cfg["inference"].update({"viewer_size": 96, "viewer_max_slices": 32})
    cfg["readers"].update({"seeds": 2, "learning_curve_n": [0, 2]})
    cfg["policy"]["min_feedback"] = 2
    return cfg


def test_end_to_end_pipeline_with_stub_models(tmp_path):
    from radagent import pipeline as P
    from radagent.store import RunStore

    cfg = _e2e_cfg(tmp_path)
    cwd = os.getcwd()
    os.chdir(ROOT)                                    # plugins are imported as tests.fixtures.*
    try:
        P.step_data(cfg)
        P.step_train(cfg)
        P.step_infer(cfg)
        P.step_vlm(cfg)
        cal = P.step_calibrate(cfg)
        P.step_first_read(cfg, "rule")
        P.step_simulate(cfg, 2)
        figs = P.step_figures(cfg)
        st = RunStore(Path(cfg["output_root"]) / "e2e")
        recs = st.records(["calib", "test"])
        assert recs and all(r["vlm"]["status"] == "ok" for r in recs)
        assert all(0.0 <= r["confidence"]["tumour_probability"] <= 1.0 for r in recs)
        assert all("vlm_probability" in r["confidence"] for r in recs)
        assert sorted(r["triage"]["rank"] for r in recs if r["split"] == "test") == \
            list(range(1, 1 + sum(r["split"] == "test" for r in recs)))
        rep = st.report_path(recs[0]["case_id"]).read_text()
        assert "SECOND READ" in rep and "stub-vlm" in rep
        calib = [bool(r["reference"]["tumour_present"]) for r in recs if r["split"] == "calib"]
        assert len(calib) == 5 and len(set(calib)) == 2   # seed 11: both classes -> the tools are compared
        assert {"segmenter", "medgemma", "fused"} <= set(cal["tools_test"])
        assert figs and all(Path(f).exists() for f in figs)
        arrs = st.load_viewer(recs[0]["case_id"])
        assert "vlm_rgb" in arrs and arrs["vlm_rgb"].ndim == 4 and arrs["vlm_rgb"].shape[-1] == 3

        case = next(r["case_id"] for r in recs if r["split"] == "test")
        rec, trace = P.run_single_case(cfg, case, "rule")                      # cached perception
        assert trace["status"] == "complete" and rec["vlm"]["source"] == "cache"
        rec2, trace2 = P.run_single_case(cfg, case, "rule", recompute=True)    # from the CT file
        assert trace2["status"] == "complete" and rec2["vlm"]["source"] == "model"
        assert [s["tool"] for s in trace2["steps"]][:5] == ["load_study", "segment_liver_tumour", "measure_findings",
                                                            "quality_check", "vlm_second_read"]
    finally:
        P.release_models()
        os.chdir(cwd)
