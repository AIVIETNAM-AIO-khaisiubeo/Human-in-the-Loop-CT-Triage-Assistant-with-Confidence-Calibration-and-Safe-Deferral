"""End-to-end pipeline steps (each step is also a CLI sub-command)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from radagent.agent.planner import RulePlanner, make_planner
from radagent.agent.runner import run_agent
from radagent.agent.tools import CaseContext, load_study, measure_findings, segment_liver_tumour
from radagent.calibration import calibration_report, fit_calibrator, reliability_table
from radagent.config import run_dir, save_config
from radagent.store import RunStore, dump_json, load_json
from radagent.triage import rank_worklist


def _store(cfg) -> RunStore:
    s = RunStore(run_dir(cfg))
    s.root.mkdir(parents=True, exist_ok=True)
    return s


# --------------------------------------------------------------------------- data
def step_data(cfg: dict) -> pd.DataFrame:
    from radagent.data.lits import build_manifest

    if cfg["data"]["source"] == "synthetic":
        from radagent.data.synthetic import generate_dataset

        generate_dataset(cfg["data"]["root"], cfg["data"]["n_cases"], seed=cfg["seed"])
    df = build_manifest(cfg)
    st = _store(cfg)
    st.save_manifest(df)
    save_config(cfg, st.p("config_used.yaml"))
    print(f"[data] {len(df)} cases -> {st.manifest_path} | " +
          ", ".join(f"{k}: {v}" for k, v in df["split"].value_counts().reindex(["train", "val", "calib", "test"]).fillna(0).astype(int).items()))
    return df


def step_train(cfg: dict) -> dict:
    from radagent.train import train

    st = _store(cfg)
    return train(cfg, st.manifest())


# --------------------------------------------------------------------------- perception
def _segmenter_loader(cfg):
    cache = {}

    def get():
        if "m" not in cache:
            from radagent.models import load_segmenter

            cache["m"] = load_segmenter(cfg)
        return cache["m"]

    return get


def _contexts(cfg: dict, splits: list[str], case_ids: list[str] | None = None):
    st = _store(cfg)
    man = st.manifest()
    rows = man[man["split"].isin(splits)]
    if case_ids:
        rows = rows[rows["case_id"].isin(case_ids)]
    cal = st.calibrator()
    get = _segmenter_loader(cfg)
    for _, row in rows.iterrows():
        yield CaseContext(case_id=row["case_id"], row=row.to_dict(), cfg=cfg, store=st, calibrator=cal,
                          get_segmenter=get)


def step_infer(cfg: dict, force: bool = False) -> None:
    """Run the perception tools (load -> segment -> measure) on the evaluation splits and cache them."""
    t0 = time.time()
    splits = cfg["inference"]["eval_splits"]
    ctxs = list(_contexts(cfg, splits))
    for k, ctx in enumerate(ctxs, start=1):
        ctx.force_recompute = force
        t1 = time.time()
        g = load_study(ctx)
        s = segment_liver_tumour(ctx)
        m = measure_findings(ctx)
        ref = ctx.analysis.get("reference", {})
        print(f"[infer] {k:3d}/{len(ctxs)} {ctx.case_id} ({ctx.row['split']}) {s['source']:>5} | "
              f"liver {m['liver_ml']} ml, tumour {m['tumour_ml']} ml ({m['n_lesions']} lesions) | "
              f"raw score {ctx.analysis['segmentation']['tumour_score_raw']:.3f} | "
              f"ref tumour {ref.get('tumour_present')} Dice L {ref.get('dice_liver', float('nan')):.3f} "
              f"T {ref.get('dice_tumour', float('nan')):.3f} | {time.time() - t1:.1f}s")
        del g
    print(f"[infer] done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- calibration
def _scores(st: RunStore, man: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = man.loc[man["split"] == split, "case_id"].tolist()
    p, y, keep = [], [], []
    for cid in ids:
        if not st.has_analysis(cid):
            continue
        a = st.load_analysis(cid)
        if "reference" not in a:
            continue
        p.append(a["segmentation"]["tumour_score_raw"])
        y.append(int(a["reference"]["tumour_present"]))
        keep.append(cid)
    return np.array(p, dtype=float), np.array(y, dtype=int), keep


def step_calibrate(cfg: dict) -> dict:
    st = _store(cfg)
    man = st.manifest()
    p_cal, y_cal, _ = _scores(st, man, "calib")
    p_test, y_test, _ = _scores(st, man, "test")
    if len(p_cal) == 0:
        raise RuntimeError("no analysed calibration cases - run `infer` first")
    cal, info = fit_calibrator(p_cal, y_cal, cfg["confidence"]["method"], seed=cfg["seed"])
    nb = cfg["confidence"]["n_bins"]
    out = {"calibrator": cal.to_dict(), "fit": info, "n_bins": nb}
    for split, (p, y) in {"calib": (p_cal, y_cal), "test": (p_test, y_test)}.items():
        if len(p) == 0:
            continue
        pc = cal(p)
        out[split] = {"before": calibration_report(p, y, nb, seed=cfg["seed"]),
                      "after": calibration_report(pc, y, nb, seed=cfg["seed"]),
                      "reliability_before": reliability_table(p, y, nb),
                      "reliability_after": reliability_table(pc, y, nb)}
    dump_json(cal.to_dict(), st.calibrator_path())
    dump_json(out, st.p("calibration", "calibration_metrics.json"))
    t = out.get("test", {})
    if t:
        b, a = t["before"], t["after"]
        print(f"[calibrate] {cal.describe()} (selected: {info.get('selected')}, CV NLL {info.get('cv_nll')})")
        print(f"[calibrate] TEST n={b['n']} (pos {b['n_pos']}): Brier {b['brier']:.3f} -> {a['brier']:.3f} | "
              f"ECE {b['ece']:.3f} -> {a['ece']:.3f} | NLL {b['nll']:.3f} -> {a['nll']:.3f} | AUROC {a['auroc']:.3f}")
    return out


# --------------------------------------------------------------------------- first read
def step_first_read(cfg: dict, planner_name: str | None = None) -> list[dict]:
    st = _store(cfg)
    if planner_name:
        cfg = {**cfg, "agent": {**cfg["agent"], "planner": planner_name}}
    planner = make_planner(cfg)
    if not st.calibrator_path().exists():
        print("[first-read] WARNING: no calibrator - confidences are raw scores. Run `calibrate` first.")
    records, t0 = [], time.time()
    ctxs = list(_contexts(cfg, cfg["inference"]["eval_splits"]))
    for k, ctx in enumerate(ctxs, start=1):
        ctx.record = {"case_id": ctx.case_id, "split": ctx.row["split"], "source": cfg["data"]["source"],
                      "review_status": "pending"}
        trace = run_agent(ctx, planner, cfg["agent"]["max_steps"])
        if trace["status"] != "complete":
            print(f"[first-read] {ctx.case_id}: agent status {trace['status']} - {trace['steps'][-1].get('summary')}")
            continue
        records.append(ctx.record)
        if k % 10 == 0 or k == len(ctxs):
            print(f"[first-read] {k}/{len(ctxs)} cases ({time.time() - t0:.0f}s, planner {trace['planner']}, "
                  f"overrides {trace['n_guardrail_overrides']})")
    for split in {r["split"] for r in records}:
        rank_worklist([r for r in records if r["split"] == split])
    for r in records:
        r.pop("report_text", None)
        st.save_record(r)
    # re-draft reports now that ranks are known
    from radagent.report import draft_report

    for r in records:
        st.report_path(r["case_id"]).write_text(draft_report(r))
    cats = pd.Series([r["triage"]["category"] for r in records if r["split"] == "test"]).value_counts().to_dict()
    print(f"[first-read] {len(records)} case records written | test-split triage: {cats}")
    return records


def run_single_case(cfg: dict, case_id: str, planner_name: str = "rule", reader_policy=None, save: bool = False) -> tuple[dict, dict]:
    """Re-run the agent on one (cached) case - used by the UI and the CLI demo."""
    cfg2 = {**cfg, "agent": {**cfg["agent"], "planner": planner_name}}
    ctx = next(_contexts(cfg2, ["train", "val", "calib", "test"], [case_id]))
    ctx.reader_policy = reader_policy
    ctx.persist = save
    ctx.record = {"case_id": case_id, "split": ctx.row["split"], "source": cfg["data"]["source"], "review_status": "pending"}
    planner = make_planner(cfg2) if planner_name == "llm" else RulePlanner()
    trace = run_agent(ctx, planner, cfg["agent"]["max_steps"], save=save)
    return ctx.record, trace


# --------------------------------------------------------------------------- readers
def step_simulate(cfg: dict, seeds: int | None = None) -> dict:
    from radagent.experiments import run_reader_experiment

    st = _store(cfg)
    records = st.records(["calib", "test"])
    out = run_reader_experiment(records, cfg, seeds=seeds)
    d = st.p("experiments")
    d.mkdir(parents=True, exist_ok=True)
    for k in ("results", "learning_curve", "policies", "burnin", "summary", "benefit_spread"):
        out[k].to_csv(d / f"{k}.csv", index=False)
    dump_json(out["meta"], d / "meta.json")
    s = out["summary"]
    print("[simulate] test-split means over seeds (accuracy | harmful acceptance | deferral | actions/case):")
    for _, r in s.iterrows():
        print(f"   {r['reader']:9s} {str(r['policy']):24s} acc {r['accuracy']:.3f} | harm {r['harmful_acceptance_rate']:.3f} | "
              f"defer {r['deferral_rate']:.2f} | actions {r['actions_per_case']:.2f} | thr {r['threshold']:.2f}")
    print("[simulate] benefit spread across readers (RQ1):")
    for _, r in out["benefit_spread"].iterrows():
        print(f"   {str(r['policy']):24s} mean benefit {r['mean_benefit']:+.3f} | spread {r['spread_max_minus_min']:.3f}")
    return out


def step_figures(cfg: dict) -> list[str]:
    from radagent import plots

    st = _store(cfg)
    return plots.make_all(st, cfg)


def step_all(cfg: dict) -> None:
    t0 = time.time()
    step_data(cfg)
    from radagent.models import checkpoint_path

    if not checkpoint_path(cfg).exists():
        step_train(cfg)
    else:
        print(f"[train] found {checkpoint_path(cfg)} - skipping training (delete it to retrain)")
    step_infer(cfg)
    step_calibrate(cfg)
    step_first_read(cfg)
    step_simulate(cfg)
    files = step_figures(cfg)
    print(f"[all] finished in {(time.time() - t0) / 60:.1f} min; {len(files)} figures in {run_dir(cfg) / 'figures'}")
    print(f"[all] open the app:  streamlit run app/streamlit_app.py -- --run {run_dir(cfg)}")


__all__ = ["step_data", "step_train", "step_infer", "step_calibrate", "step_first_read", "step_simulate",
           "step_figures", "step_all", "run_single_case", "load_json"]
