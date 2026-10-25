"""Command line:  python -m radagent <command> --config configs/demo.yaml

Commands
  data        generate synthetic phantoms (or index LiTS) and write the case manifest
  train       train the segmentation tool
  infer       run perception tools (segment + measure) on the calib/test cases
  calibrate   fit the confidence calibrator on the calib split, evaluate on test
  first-read  run the agent (plan -> tools -> report) on every evaluation case
  simulate    reader experiment: universal vs reader-specific vs deferral policies
  figures     make all plots
  all         everything above, in order
  agent       run the agent on one case and print its trace (try --planner llm)
"""
from __future__ import annotations

import argparse
import json

from radagent.config import load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="radagent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["data", "train", "infer", "calibrate", "first-read", "simulate", "figures",
                                        "all", "agent"])
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--force", action="store_true", help="infer: recompute cached analyses")
    ap.add_argument("--seeds", type=int, default=None, help="simulate: number of seeds")
    ap.add_argument("--planner", choices=["rule", "llm"], default=None)
    ap.add_argument("--case", default=None, help="agent: case id (default: first test case)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    from radagent import pipeline as P

    if args.command == "data":
        P.step_data(cfg)
    elif args.command == "train":
        P.step_train(cfg)
    elif args.command == "infer":
        P.step_infer(cfg, force=args.force)
    elif args.command == "calibrate":
        P.step_calibrate(cfg)
    elif args.command == "first-read":
        P.step_first_read(cfg, args.planner)
    elif args.command == "simulate":
        P.step_simulate(cfg, args.seeds)
    elif args.command == "figures":
        for f in P.step_figures(cfg):
            print(f)
    elif args.command == "all":
        P.step_all(cfg)
    elif args.command == "agent":
        from radagent.store import RunStore
        from radagent.config import run_dir

        man = RunStore(run_dir(cfg)).manifest()
        case = args.case or man.loc[man["split"] == "test", "case_id"].iloc[0]
        rec, trace = P.run_single_case(cfg, case, args.planner or cfg["agent"]["planner"])
        for s in trace["steps"]:
            g = f"  [guardrail: {s['guardrail']}]" if s.get("guardrail") else ""
            print(f"{s['step']:2d}. {s['tool']:22s} by {s['planned_by']:26s} {s['duration_ms']:8.1f} ms  "
                  f"{json.dumps(s['summary'], default=str)[:110]}{g}")
            if s.get("reason"):
                print(f"      reason: {s['reason']}")
        print(f"status: {trace['status']}")


if __name__ == "__main__":
    main()
