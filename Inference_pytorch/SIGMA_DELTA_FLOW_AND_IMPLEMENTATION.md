# Sigma-Delta (ΣΔ) Stream CiM: Flow and Implementation

This document describes how **stream-based ΣΔ** activation I/O is modeled in this NeuroSim fork, how it differs from the **baseline SAR-ADC** path, and where the code lives.

---

## 1. Conceptual flow (what we model)

### Baseline (ADC) path

1. **Analog MAC / array read** produces a partial result per subarray / timing window.
2. **SAR ADC** (or equivalent) quantizes the analog value to a digital code each read cycle.
3. **Digital periphery** (PE → tile → chip) performs **accumulation**, **buffering**, **interconnect (H-tree / bus)**, **pooling**, and **activation** blocks as separate modeled components with area, latency, and energy.

In NeuroSim, these blocks appear explicitly in layer/chip summaries (buffer latency, IC latency, accumulation latency, etc.).

### Sigma-delta stream path (this project)

1. The **same physical read / CiM** still produces a signal that must be **resolved in time**; we replace the **SAR ADC block** with a **ΣΔ modulator** model (`SigmaDeltaModulator` / row variant) when `cimInterfaceMode == SIGMA_DELTA_STREAM`.
2. **Information is encoded in the time domain**: a 1-bit stream representation whose **time-average** tracks the analog value over an **observation window** \(T_o\), with a **natural frequency** \(f_c\) of modulator and analog parameters (e.g. \(C_\mathrm{int}\), \(I_\mathrm{ref}\), \(V_\mathrm{dd}\)) used for modulator PPA.
3. **Periphery accounting**: aligned with the EAS-CiM style story—**no separate digital accumulation trees, large SRAM buffers, or heavy global interconnect** as *first-class* contributors in ΣΔ mode for the main datapath. Those paths are **gated off** at PE, tile, and chip so their **latency and dynamic energy are zero** for buffer / accumulation / bus-style blocks (subarray-level breakdown may still show energy under generic labels where ΣΔ energy is bucketed). **Max-pooling and chip-level activation** are handled separately: by default in ΣΔ mode they use an **analog stream model** (integrate-and-compare style comparator budgeting over \(T_o\); see §7). Set **`EASCIM_ANALOG_POOL_ACT=0`** to fall back to the **digital** `MaxPooling` / activation PPA instead.

So: **ΣΔ = swap readout interface + bypass most modeled digital post-processing hierarchy**, with **pool + activation** either as a lightweight analog-style counter model or full digital blocks, not a change to the CNN math in PyTorch.

### Physical picture (first-order ΣΔ)

A simplified first-order loop contains:

1. **Integrator** on capacitor \(C_{\mathrm{int}}\) driven by the **difference** between input current (proportional to the quantity to encode) and **feedback** current switched according to the 1-bit quantizer output.
2. **Comparator / hysteresis stage** (in the paper: DLS inverter) deciding when the integrator has crossed a threshold band.
3. **1-bit DAC / feedback** that injects \(\pm I_{\mathrm{ref}}\) (or equivalent) back onto \(C_{\mathrm{int}}\).

Over a finite **observation window** \(T_o\), the **average duty cycle** \(\delta\) of the output bit stream relates to the normalized input (paper Eqs. (1)–(2) in the EAS-CiM 2.0 manuscript). A higher **natural clocking rate** \(f_c\) (internal ring-oscillator / DLS activity) allows more **edges per \(T_o\)**, improving **effective resolution** at the cost of **dynamic energy** (more switching on \(C_{\mathrm{int}}\)).

In NeuroSim, **`SigmaDeltaModulator`** (`SigmaDeltaModulator.h` / `.cpp`) encapsulates:

- **Roles:** `OUTPUT_ENCODER` (readout stream after array) vs `INPUT_ENCODER` (row / WL side, when instantiated as `sigmaDeltaModulatorRow`).
- **Knobs:** `naturalFreqFc` (\(f_c\)), `observationPeriodTo` (\(T_o\)), `cIntFemtoFarad`, `iRefNanoAmp`, plus `eascimSigmaDeltaVdd` on `Param` for switching-energy scaling.


## 2. Where it is implemented (code map)

| Layer | File | Role |
|--------|------|------|
| **Configuration** | `NeuroSIM/Param.cpp` | `CiMInterfaceMode` (`BASELINE_ADC` vs `SIGMA_DELTA_STREAM`), ΣΔ knobs (`eascimNaturalFreqFc`, `eascimObservationPeriodTo`, …). **Env overrides** for sweeps: `NS_CIM_INTERFACE_MODE`, `EASCIM_FC_HZ`, `EASCIM_TO_S`, `EASCIM_ANALOG_POOL_ACT`, `EASCIM_ANALOG_E_PJ`, etc. |
| **Subarray readout** | `NeuroSIM/SubArray.cpp` | Chooses SAR ADC vs ΣΔ modulator based on `cimInterfaceMode`; ΣΔ area/latency/power instead of ADC when in stream mode. |
| **PE** | `NeuroSIM/ProcessingUnit.cpp` | If ΣΔ: skip adder tree, buffer, internal bus PPA aggregation for the “digital activation path”; performance uses subarray timing only. |
| **Tile** | `NeuroSIM/Tile.cpp` | If ΣΔ: skip tile accumulation, buffers, ReLU/Sigmoid, H-tree timing/energy for that path. |
| **Chip** | `NeuroSIM/Chip.cpp` | If ΣΔ: skip global buffer and global accumulation timing/energy. **Max-pool + chip activation**: analog stream helper (`AnalogStreamNonlinearity.*`) when `eascimAnalogPoolActivate` is true; else digital `MaxPooling` / `GreLu` / `Gsigmoid` PPA. |
| **PyTorch → NeuroSim** | `Inference_pytorch/inference.py` | `--inference 1` triggers hooks → CSV traces → `main`; `num_workers=0` for stable DataLoader; `python -u` in scripts for live logs. |
| **Sweeps / logs** | `Inference_pytorch/scripts/run_cs6501_project.sh` | Activates **`neurosim`** conda env, runs baseline + ΣΔ grids with env vars. |
| **Summaries** | `scripts/parse_project_results.py` | Parses per-run `*.log` → `summary.csv` (accuracy, clk, pipeline latency, energy, TOPS/W, FPS, area). |
| **Overleaf** | `scripts/make_overleaf_results.py` | Writes **`overleaf_results.tex` only** (tables: baseline vs ΣΔ at reference \((T_o,f_c)\), plus full ΣΔ sweep table). No PDF generation. |

---

## 3. How a run flows end-to-end

1. **Environment** (recommended): `conda activate neurosim` so PyTorch/CUDA match the machine used for pretrained checkpoints.
2. **`inference.py`** loads CIFAR-10, VGG8 (or other model), runs **one test pass** with hooks on the first batch when `--inference 1`.
3. **Hooks** write layer traces and invoke the NeuroSim C++ backend (`main`) with the current `Param` (including CiM mode and ΣΔ parameters from `Param.cpp`, possibly overridden by env).
4. **NeuroSim** builds floorplan, then for each layer reports read latency/energy and breakdowns; at chip level reports pipeline metrics, TOPS/W, FPS, area.
5. **ΣΔ vs baseline** is selected entirely by **`NS_CIM_INTERFACE_MODE`** (`0` = ADC path with full digital periphery; `1` = ΣΔ readout + bypassed hierarchy as above).

---

## 4. Parameters you sweep (experiment side)

Typical sweep grid (as used in project logs):

- **`NS_CIM_INTERFACE_MODE`**: `0` baseline, `1` ΣΔ.
- **`EASCIM_TO_S`**: observation window \(T_o\) (e.g. `1e-9`, `5e-9`, `1e-8`).
- **`EASCIM_FC_HZ`**: natural frequency \(f_c\) of modulator (e.g. `50e6`, `100e6`, `200e6`).
- **`EASCIM_SD_VDD_V`**: ΣΔ supply (e.g. `0.4`).
- **`subArray` / `parallelRead`**: must satisfy `parallelRead == subArray` when `cellBit > 1` (multi-level cell constraint in `inference.py`).

Results are aggregated with **`parse_project_results.py`** and exported for Overleaf with **`make_overleaf_results.py`**.

---

## 5. How to reproduce the final summary folder

From `Inference_pytorch/`:

```bash
source /path/to/conda.sh && conda activate neurosim
python3 scripts/parse_project_results.py logrun2/sweep_<timestamp>
python3 scripts/make_overleaf_results.py logrun2/sweep_<timestamp>
```

The deliverable is **`logrun2/sweep_<timestamp>/overleaf_results.tex`** plus **`summary.csv`** in the same directory.

---

## 6. Relation to the papers

The implementation is meant to reflect **stream-based, time-encoded readout** and **reduced reliance on conventional digital accumulation / buffering / activation** along the PE–tile–chip path, as discussed in the EAS-CiM / EASI-CiM papers, not a cycle-accurate SPICE model of every transistor, but a **consistent architectural accounting** inside NeuroSim’s existing PPA framework.

---

## 7. Analog stream max-pool and activation (optional PPA)

When **`cimInterfaceMode == SIGMA_DELTA_STREAM`** and **`eascimAnalogPoolActivate`** is true (default for ΣΔ unless overridden), `Chip.cpp` routes **max-pool area** and **chip ReLU/sigmoid area, latency, and dynamic energy** through **`AnalogStreamNonlinearity`** (`NeuroSIM/AnalogStreamNonlinearity.h/.cpp`). The model is **architectural** (comparator counts tied to \(T_o\), \(f_c\), pool window, and MPU-style tree depth; tunable energy/area factors), **not** extracted layout or SPICE.

**Environment variables**

- **`EASCIM_ANALOG_POOL_ACT`**: `0` = use digital max-pool / activation blocks under ΣΔ; `1` = use analog stream model (only meaningful when interface is ΣΔ).
- **`EASCIM_ANALOG_E_PJ`**: optional override for energy per compare event (pJ); other β / stream factors live on `Param` (`eascimAnalogEnergyPerComparePJ`, `eascimAnalogCompAreaBeta`, `eascimAnalogStreamEnergyFactor` in `Param.h` / defaults in `Param.cpp`).

`main` prints **`analogPoolActivate`** under the ΣΔ banner so logs show which branch ran.

---

