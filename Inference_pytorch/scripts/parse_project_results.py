#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Record:
    run: str
    model: str
    mode: str  # baseline|sigmadelta
    subarray: int
    to_s: Optional[float]
    fc_hz: Optional[float]
    acc_pct: Optional[float]
    clk_ns: Optional[float]
    pipe_ns: Optional[float]
    dyn_pj: Optional[float]
    topsw: Optional[float]
    fps: Optional[float]
    chip_area_um2: Optional[float]


def ffloat(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_log(path: Path) -> Record:
    txt = path.read_text(errors="ignore")
    run = path.stem

    # infer tags from filename convention in run script
    model = "VGG8" if "VGG8" in run else ("DenseNet40" if "DenseNet40" in run else "unknown")
    mode = "sigmadelta" if "sigmadelta" in run else "baseline"

    m_sub = re.search(r"_sub(\d+)", run)
    sub = int(m_sub.group(1)) if m_sub else -1

    m_to = re.search(r"_To([0-9eE\-\.+]+)", run)
    to_s = ffloat(m_to.group(1)) if m_to else None
    m_fc = re.search(r"_fc([0-9eE\-\.+]+)", run)
    fc_hz = ffloat(m_fc.group(1)) if m_fc else None

    m_acc = re.search(r"Accuracy:\s*\d+/\d+\s*\((\d+)\%\)", txt)
    acc = float(m_acc.group(1)) if m_acc else None

    m_clk = re.search(r"Chip clock period is:\s*([0-9eE\-\.+]+)ns", txt)
    clk_ns = ffloat(m_clk.group(1)) if m_clk else None

    m_pipe = re.search(r"Chip pipeline-system-clock-cycle \(per image\) is:\s*([0-9eE\-\.+]+)ns", txt)
    pipe_ns = ffloat(m_pipe.group(1)) if m_pipe else None

    m_dyn = re.search(r"Chip pipeline-system readDynamicEnergy \(per image\) is:\s*([0-9eE\-\.+]+)pJ", txt)
    dyn_pj = ffloat(m_dyn.group(1)) if m_dyn else None

    m_topsw = re.search(r"Energy Efficiency TOPS/W \(Pipelined Process\):\s*([0-9eE\-\.+]+)", txt)
    topsw = ffloat(m_topsw.group(1)) if m_topsw else None

    m_fps = re.search(r"Throughput FPS \(Pipelined Process\):\s*([0-9eE\-\.+]+)", txt)
    fps = ffloat(m_fps.group(1)) if m_fps else None

    m_area = re.search(r"ChipArea\s*:\s*([0-9eE\-\.+]+)um\^2", txt)
    area_um2 = ffloat(m_area.group(1)) if m_area else None

    return Record(
        run=run,
        model=model,
        mode=mode,
        subarray=sub,
        to_s=to_s,
        fc_hz=fc_hz,
        acc_pct=acc,
        clk_ns=clk_ns,
        pipe_ns=pipe_ns,
        dyn_pj=dyn_pj,
        topsw=topsw,
        fps=fps,
        chip_area_um2=area_um2,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_dir",
        type=Path,
        help=(
            "Either project_results/<timestamp> directory (expects ./logs/*.log) "
            "or a sweep directory (e.g., logrun2/sweep_<ts>) containing *.log files."
        ),
    )
    args = ap.parse_args()

    # Support both layouts:
    # - project_results/<ts>/logs/*.log
    # - logrun2/sweep_<ts>/*.log (and optionally nested logs)
    logs_dir = args.results_dir / "logs"
    if logs_dir.is_dir():
        log_paths = sorted(logs_dir.glob("*.log"))
    else:
        log_paths = sorted(args.results_dir.rglob("*.log"))

    out_csv = args.results_dir / "summary.csv"

    rows = []
    for p in log_paths:
        if p.name in ("build_info.log", "master.log", "master_sweep.log"):
            continue
        rows.append(parse_log(p))

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    print(f"Wrote {len(rows)} rows to {out_csv}")
    if not rows:
        print("No logs parsed. Expected *.log files with NeuroSim summary lines.")


if __name__ == "__main__":
    main()

