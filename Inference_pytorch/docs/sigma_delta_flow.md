# Sigma-delta (ΣΔ) stream interface in NeuroSim (`Inference_pytorch`)

**NeuroSim extension notes (EAS-CiM 2.0–aligned)**

This document explains how **first-order sigma–delta (ΣΔ) modulation** is modeled in the `Inference_pytorch/NeuroSIM` C++ path when the compute-in-memory (CiM) interface is set to **event-driven / stream-style readout**, following the spirit of **EAS-CiM 2.0** (Sreekumar *et al.*, ISCAS 2025): a **1-bit feedback stream** whose **duty cycle** and **effective pulse rate** encode an analog quantity (here, accumulated column current / MAC result) over an **observation window** \(T_o\).

> **Scope:** This is a *PPA-oriented architectural model* in NeuroSim, not a SPICE netlist. Knobs (\(f_c\), \(T_o\), \(C_{\mathrm{int}}\), \(I_{\mathrm{ref}}\), ΣΔ supply \(V_{\mathrm{dd}}\)) calibrate area, latency, and energy against the paper’s ranges and your technology node.

## 1. Where the mode is selected

`Param::cimInterfaceMode` chooses the readout accounting path:

| Mode | Meaning |
|------|---------|
| `BASELINE_ADC` | Classic NeuroSim path: SAR-ADC or MLSA, depending on existing `SARADC` / MLSA flags. |
| `SIGMA_DELTA_STREAM` | ΣΔ **output encoder** (column / sense-line side) replaces SAR latency/energy/area hooks in the subarray macros. |

Row-side ΣΔ (word-line / activation encoding) is **modeled for timing** when the cell type is RRAM or FeFET; **separate area/energy accounting** for that row block is optional via `eascimRowSigmaDeltaSeparateAccounting`.

## 2. Physical picture (first-order ΣΔ)

A simplified first-order loop contains:

1. **Integrator** on capacitor \(C_{\mathrm{int}}\) driven by the **difference** between input current (proportional to the quantity to encode) and **feedback** current switched according to the 1-bit quantizer output.
2. **Comparator / hysteresis stage** (in the paper: DLS inverter) deciding when the integrator has crossed a threshold band.
3. **1-bit DAC / feedback** that injects \(\pm I_{\mathrm{ref}}\) (or equivalent) back onto \(C_{\mathrm{int}}\).

Over a finite **observation window** \(T_o\), the **average duty cycle** \(\delta\) of the output bit stream relates to the normalized input (paper Eqs. (1)–(2) in the EAS-CiM 2.0 manuscript). A higher **natural clocking rate** \(f_c\) (internal ring-oscillator / DLS activity) allows more **edges per \(T_o\)**, improving **effective resolution** at the cost of **dynamic energy** (more switching on \(C_{\mathrm{int}}\)).

In NeuroSim, **`SigmaDeltaModulator`** (`SigmaDeltaModulator.h` / `.cpp`) encapsulates:

- **Roles:** `OUTPUT_ENCODER` (readout stream after array) vs `INPUT_ENCODER` (row / WL side, when instantiated as `sigmaDeltaModulatorRow`).
- **Knobs:** `naturalFreqFc` (\(f_c\)), `observationPeriodTo` (\(T_o\)), `cIntFemtoFarad`, `iRefNanoAmp`, plus `eascimSigmaDeltaVdd` on `Param` for switching-energy scaling.

## 3. Data flow through NeuroSim (high level)

```
  models/*.py, modules/*.py, utee/hook.py
              |
              v
     layer_record_*  (CSV weights / activations)
              |
              v
     NeuroSIM main.cpp + Param
              |
              v
          SubArray  ------>  sigmaDeltaModulator (OUTPUT_ENCODER)
              |
              +--(RRAM/FeFET)-->  sigmaDeltaModulatorRow (INPUT_ENCODER), timing optional in PPA
```

1. **PyTorch inference** exports per-layer tensors to CSV under `layer_record_*` (weights, activations).
2. **`main`** reads `Param` (defaults in `Param.cpp`), builds tiles / subarrays, and runs latency–area–energy calculators.
3. **`SubArray`** selects SAR vs ΣΔ via macros (`NS_USE_SIGMA_DELTA`, `NS_SUBARRAY_ADC_*`). In ΣΔ mode, **`sigmaDeltaModulator`** is sized, timed, and energized like other `FunctionUnit` peripherals.

## 4. Latency model (what enters `readLatency`)

For the output ΣΔ block, after `Initialize`, **`CalculateLatency(numRead)`** sets:

\[
\tau_{\Sigma\Delta,\mathrm{out}} \;=\; T_o \times \texttt{numRead}
\]

That is **`observationPeriodTo * numRead`** (seconds in internal SI units), i.e. **one full observation window per counted read operation** in the NeuroSim accounting path. There is **no extra explicit “settle” term** folded into this line in the current implementation; any settling behavior is implicitly folded into how you set \(T_o\) and \(f_c\) for your study.

**Row-side** ΣΔ (`sigmaDeltaModulatorRow`) uses the same `CalculateLatency` pattern when the row modulator is present, so **`readLatency` on the subarray can include both** output and row contributions depending on the `NS_SUBROW_SDM_RL` macro path.

## 5. `readLatency` vs `readLatencySync` (global clock proxy)

NeuroSim derives a **chip-level synchronous clock period** from the **maximum** subarray `readLatency` seen along critical paths (`ProcessingUnit.cpp`). A long \(T_o\)-dominated ΣΔ observation time can therefore **inflate the global `clkPeriod`** and depress reported **FPS**, even if the physical array could be clocked faster in a fully asynchronous SoC.

To separate **“bit-serial ΣΔ observation time”** from **“array / digital synchronous cycle”** for exploration, `SubArray` exposes **`readLatencySync`**:

- **`readLatency`:** full modeled read path latency (includes output ΣΔ \(T_o \cdot \texttt{numRead}\), and row ΣΔ timing when modeled).
- **`readLatencySync`:** same as `readLatency`, then **minus output ΣΔ `sigmaDeltaModulator.readLatency` only** when `SIGMA_DELTA_STREAM` is active (row ΣΔ is **not** subtracted here).

**`ProcessingUnit`** uses **`readLatencySync`** when updating `clkPeriod`, so the **global clock / FPS** reflects a **decoupled** (synchronous-equivalent) view while **energy and detailed read latency** can still include ΣΔ.

Interpretation: use **`readLatency`** for **end-to-end read timing** where ΣΔ observation is on the critical path; use **`readLatencySync`** when you want **FPS / global clock** to ignore the output ΣΔ window (e.g. asynchronous periphery running on its own time base).

## 6. Energy and area (short)

- **Area:** transistor strip (scaled from 65 nm reference) + MiM bank \(\propto C_{\mathrm{int}}\) via `eascimAreaMimPerFfM2` and `eascimAreaOverheadFactor`.
- **Dynamic energy (output path):** per-column contribution combines **switching on \(C_{\mathrm{int}}\)** (\(\propto\) pulse count \(\approx f_c T_o\)) and **bias / \(I_{\mathrm{ref}}\)** over \(T_o\), scaled by `eascimEnergySwitchFactor`, `eascimEnergyBiasFactor`, column resistance shaping, and temperature.

See `SigmaDeltaModulator::GetReadPathEnergy` / `GetInputPathEnergy` for the exact expressions.

## 7. Parameters to know (`Param`)

| Field | Role |
|-------|------|
| `cimInterfaceMode` | `BASELINE_ADC` vs `SIGMA_DELTA_STREAM`. |
| `eascimNaturalFreqFc` | \(f_c\) [Hz]. |
| `eascimObservationPeriodTo` | \(T_o\) [s]. |
| `eascimCintFemtoFarad` | \(C_{\mathrm{int}}\) [fF], clamped to paper range in `SigmaDeltaModulator::Initialize`. |
| `eascimIrefNanoAmp` | \(I_{\mathrm{ref}}\) [nA], clamped similarly. |
| `eascimSigmaDeltaVdd` | Dedicated ΣΔ supply for energy (e.g. 0.4 V in 65 nm style studies). |
| `eascimRowSigmaDeltaSeparateAccounting` | If `true`, count row ΣΔ area/energy as its own bucket; if `false`, row timing may still exist without separate PPA breakout. |

## 8. Logs referenced in this work

Example artifacts (paths under `Inference_pytorch/`):

- **`logrun/VGGcifar10Compare.log`** — side-by-side or comparative CIFAR-10 / VGG-style NeuroSim runs (baseline vs ΣΔ-oriented settings), useful for **Δ%** tables in R or a spreadsheet.
- **`logrun2/cifar10_vgg8_sigmadelta_paperDefaults_Vdd0p4_rowModeledNotCounted_clkDecoupled.log`** — VGG8, paper-default ΣΔ knobs, \(V_{\mathrm{dd}}=0.4\) V for the ΣΔ block, row ΣΔ **modeled but not separately counted** in PPA, **`readLatencySync`** used for global clock (decoupled ΣΔ observation from `clkPeriod`).

## References

- R. Sreekumar *et al.*, “EAS-CiM 2.0: Event-driven Asynchronous Stream-based Compute-in-Memory Kernels with Scalable Precision,” *IEEE ISCAS*, 2025.
- Original NeuroSim: P.-Y. Chen, X. Peng, S. Yu, Arizona State University (see file headers in `NeuroSIM/`).
