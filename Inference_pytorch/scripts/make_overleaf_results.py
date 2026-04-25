#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def latex_table_baseline_vs_sigmadelta(rows: List[Dict[str, Any]], model: str, subarray: int) -> str:
    # pick the first baseline run for this model+subarray and the first sigmadelta run with To=5e-9 fc=100e6
    base = None
    sd = None
    for r in rows:
        if r["model"] == model and int(float(r["subarray"])) == subarray and r["mode"] == "baseline":
            base = r
            break
    for r in rows:
        if r["model"] == model and int(float(r["subarray"])) == subarray and r["mode"] == "sigmadelta":
            if abs(f(r.get("to_s", "nan")) - 5e-9) < 1e-15 and abs(f(r.get("fc_hz", "nan")) - 100e6) < 1:
                sd = r
                break
    if base is None or sd is None:
        return "% Table skipped: missing baseline or sigma-delta reference point\n"

    def fmt_num(x: str, scale: float = 1.0) -> str:
        v = f(x) * scale
        if v != v:
            return "--"
        if abs(v) >= 1e6 or abs(v) < 1e-2:
            return f"{v:.2e}"
        return f"{v:.3g}"

    # Change ratios: sd/base
    def ratio(sd_val: str, b_val: str) -> str:
        s = f(sd_val); b = f(b_val)
        if s != s or b != b or b == 0:
            return "--"
        return f"{(s/b):.2f}×"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Baseline vs. ΣΔ (reference point $T_o=5\\,\\mathrm{{ns}}$, $f_c=100\\,\\mathrm{{MHz}}$) for {model} (subArray={subarray}).}}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Metric & Baseline & ΣΔ & Change (ΣΔ/B)\\\\")
    lines.append("\\midrule")
    lines.append(f"Clock period (ns) & {fmt_num(base['clk_ns'])} & {fmt_num(sd['clk_ns'])} & {ratio(sd['clk_ns'], base['clk_ns'])}\\\\")
    lines.append(f"Pipeline latency (ns/img) & {fmt_num(base['pipe_ns'])} & {fmt_num(sd['pipe_ns'])} & {ratio(sd['pipe_ns'], base['pipe_ns'])}\\\\")
    lines.append(f"Dynamic energy (µJ/img) & {fmt_num(base['dyn_pj'], 1e-6)} & {fmt_num(sd['dyn_pj'], 1e-6)} & {ratio(sd['dyn_pj'], base['dyn_pj'])}\\\\")
    lines.append(f"Energy efficiency (TOPS/W) & {fmt_num(base['topsw'])} & {fmt_num(sd['topsw'])} & {ratio(sd['topsw'], base['topsw'])}\\\\")
    lines.append(f"Throughput (FPS) & {fmt_num(base['fps'])} & {fmt_num(sd['fps'])} & {ratio(sd['fps'], base['fps'])}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def latex_table_sigmadelta_sweep(rows: List[Dict[str, Any]], model: str, subarray: int) -> str:
    pts: List[Tuple[float, float, Dict[str, Any]]] = []
    for r in rows:
        if r.get("model") != model:
            continue
        if r.get("mode") != "sigmadelta":
            continue
        if int(float(r.get("subarray", "nan"))) != subarray:
            continue
        to_s = f(r.get("to_s", "nan"))
        fc_hz = f(r.get("fc_hz", "nan"))
        if to_s != to_s or fc_hz != fc_hz:
            continue
        pts.append((to_s, fc_hz, r))

    if not pts:
        return "% Table skipped: no sigma-delta sweep points found\n"

    pts.sort(key=lambda t: (t[0], t[1]))

    def fmt(x: str, scale: float = 1.0) -> str:
        v = f(x) * scale
        if v != v:
            return "--"
        if abs(v) >= 1e6 or abs(v) < 1e-2:
            return f"{v:.2e}"
        return f"{v:.4g}"

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{ΣΔ sweep results for {model} (subArray={subarray}, parallelRead={subarray}).}}")
    lines.append("\\begin{tabular}{rrrrrr}")
    lines.append("\\toprule")
    lines.append("$T_o$ (ns) & $f_c$ (MHz) & clk (ns) & pipe (ns/img) & dyn (\\(\\mu\\)J/img) & FPS\\\\")
    lines.append("\\midrule")
    for to_s, fc_hz, r in pts:
        lines.append(
            f"{to_s*1e9:.3g} & {fc_hz/1e6:.0f} & {fmt(r.get('clk_ns','nan'))} & {fmt(r.get('pipe_ns','nan'))} & {fmt(r.get('dyn_pj','nan'),1e-6)} & {fmt(r.get('fps','nan'))}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing summary.csv (project_results/<ts> or logrun2/sweep_<ts>)",
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    rows = load_csv(results_dir / "summary.csv")

    tex = results_dir / "overleaf_results.tex"
    with tex.open("w") as f:
        f.write("% Auto-generated results snippet (Overleaf-ready)\n")
        f.write("% Requires: \\usepackage{booktabs}\n\n")
        f.write("\\section{Results}\n")
        f.write("\\subsection{Baseline vs. ΣΔ interface}\n")
        f.write(latex_table_baseline_vs_sigmadelta(rows, "VGG8", 128))
        f.write(latex_table_baseline_vs_sigmadelta(rows, "DenseNet40", 128))
        f.write("\\subsection{ΣΔ sweep table}\n")
        f.write(latex_table_sigmadelta_sweep(rows, "VGG8", 128))

    print(f"Wrote LaTeX snippet: {tex}")


if __name__ == "__main__":
    main()

