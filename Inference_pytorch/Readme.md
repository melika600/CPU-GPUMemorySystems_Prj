# Sigma-Delta (ΣΔ) Stream CiM: Flow and Implementation

This document describes how **stream-based ΣΔ** activation I/O is modeled in this NeuroSim fork, how it differs from the **baseline SAR-ADC** path, and where the code lives.


## Inference_pytorch repository layout (this fork)

This repo includes an `Inference_pytorch/` folder that contains the PyTorch wrapper plus this fork’s **Sigma-Delta (ΣΔ) stream CiM** extensions and experiment artifacts.

- **`Inference_pytorch/finallog/`**: Results for baseline vs ΣΔ experiments with VGG8 and the sweeps of fc (e.g. `master.log`, per-run `logs/*.log`, `summary.csv`).
- **`Inference_pytorch/finallog2/`**: Results for baseline vs ΣΔ experiments with DenseNet40 and the sweeps of fc (same structure as `finallog/`).
- **`Inference_pytorch/finallog_npulses/`**: Results for ΣΔ experiments with VGG8 that sweep an **effective pulse count** \(N\) (you’ll see `..._N{1,3,5,10,20}_...` in filenames).
- **`Inference_pytorch/log/`**: Runtime artifacts and model checkpoints used by the wrapper (e.g. `VGG8.pth`, `DenseNet40.pth`) plus any default log outputs that aren’t part of the curated `finallog*` bundles.
- **`Inference_pytorch/logrun/`**: Small, tracked benchmark/reference log(s) used for quick comparisons (the rest of run logs are typically ignored by `.gitignore`).
- **`Inference_pytorch/models/`**: PyTorch model definitions (e.g. VGG / ResNet / DenseNet) and dataset helpers.
- **`Inference_pytorch/modules/`**: Quantization / inference helper modules used by the wrapper.
- **`Inference_pytorch/NeuroSIM/`**: C++ NeuroSim source used by the wrapper (compile with `make`). This fork includes the ΣΔ interface model (`SigmaDeltaModulator.*`) and optional analog stream pool/activation PPA (`AnalogStreamNonlinearity.*`) which are our contributions.
- **`Inference_pytorch/scripts/`**: Experiment scripts and post-processing:
  - `run_cs6501_project.sh`: example sweep runner (baseline + ΣΔ via env vars)
  - `parse_project_results.py`: parse logs → `summary.csv`
- **`Inference_pytorch/utee/`**: Utility code used by training/inference (including hooks that export layer traces for NeuroSim).
- **`Inference_pytorch/inference.py`**: Main PyTorch entrypoint for inference + NeuroSim hardware evaluation (`--inference 1` enables trace export + C++ backend run).

For the detailed ΣΔ stream flow/implementation notes, read the rest of the Readme:
  
---

## 1. Conceptual flow (what we model)

### Baseline (ADC) path

1. **Analog MAC / array read** produces a partial result per subarray / timing window.
2. **SAR ADC** (or equivalent) quantizes the analog value to a digital code each read cycle.
3. **Digital periphery** (PE → tile → chip) performs **accumulation**, **buffering**, **interconnect (H-tree / bus)**, **pooling**, and **activation** blocks as separate modeled components with area, latency, and energy.

In NeuroSim, these blocks appear explicitly in layer/chip summaries (buffer latency, IC latency, accumulation latency, etc.).

### Sigma-delta stream path (this project)

1. The **same physical read / CiM** still produces an analog quantity that must be **resolved in time**; we replace the **SAR ADC block** with a **ΣΔ modulator** model (`SigmaDeltaModulator`) when `cimInterfaceMode == SIGMA_DELTA_STREAM` (column/output-side).
2. **Information is encoded in the time domain**: a 1-bit (stream) representation whose **time-average** tracks the analog value over a finite **observation window** \(T_o\), with a **natural stream frequency** \(f_c\) and analog parameters (e.g. \(C_\mathrm{int}\), \(I_\mathrm{ref}\), \(V_\mathrm{dd}\)) used for ΣΔ PPA.
3. **Periphery accounting / gating**: aligned with the EAS-CiM / EASI-CiM story, the ΣΔ mode primarily **accounts for the readout/interface cost** and **bypasses/gates off** most of the baseline digital post-processing hierarchy along the PE–tile–chip datapath (so buffer/accumulation/bus-style contributions are not counted as in the baseline first-order model).  
   **Max-pooling and chip-level activation** are handled separately: by default in ΣΔ mode they can use an **analog stream comparator** PPA model (`AnalogStreamNonlinearity`), controlled by `EASCIM_ANALOG_POOL_ACT`. Setting `EASCIM_ANALOG_POOL_ACT=0` falls back to the original digital `MaxPooling` / activation PPA instead.

So: **ΣΔ = swap readout interface + bypass most modeled digital post-processing hierarchy**, with **pool + activation** modeled either as a lightweight analog-style comparator model or via the original digital blocks (CNN math in PyTorch is unchanged).

---

## 2. Physical picture (first-order ΣΔ)

A simplified first-order loop contains:

1. **Integrator** on capacitor \(C_{\mathrm{int}}\) driven by the difference between input current and feedback current switched by the 1-bit quantizer output.
2. **Comparator / hysteresis** stage (in the paper: DLS inverter) deciding when the integrator crosses a threshold band.
3. **1-bit DAC / feedback** injecting feedback proportional to \(\pm I_{\mathrm{ref}}\) (or equivalent) back onto the integrator node.

Over a finite **observation window** \(T_o\), the stream’s **average duty cycle** relates to the normalized input (paper Eqs. (1)–(2) in the EAS-CiM 2.0 manuscript). A higher internal clocking rate \(f_c\) produces more effective stream transitions within \(T_o\), improving effective resolution at the cost of higher switching activity (dynamic energy).

In this fork, `SigmaDeltaModulator` (`SigmaDeltaModulator.h` / `.cpp`) encapsulates:

- **Roles**
  - `OUTPUT_ENCODER`: readout stream after the array (used as `sigmaDeltaModulator`).
  - `INPUT_ENCODER`: row / word-line side when instantiated as `sigmaDeltaModulatorRow` (used for timing when the memory cell type supports it).
- **Knobs (from `Param`)**
  - `eascimNaturalFreqFc` (\(f_c\)),
  - `eascimObservationPeriodTo` (\(T_o\)),
  - `eascimCintFemtoFarad` (\(C_\mathrm{int}\)),
  - `eascimIrefNanoAmp` (\(I_\mathrm{ref}\)),
  - `eascimSigmaDeltaVdd` (\(V_\mathrm{dd}\)) for ΣΔ switching-energy scaling.
- **Area**
  - transistor budget (scaled from a 65 nm reference feature size in code comments) + MiM capacitor area scaling with \(C_\mathrm{int}\) using:
    - `eascimAreaMimPerFfM2`
    - `eascimAreaOverheadFactor`
- **Dynamic energy / power**
  - switching-energy component tied to an effective pulse count \( \approx f_c \cdot T_o \) (via `GetPulseCountInObservationWindow()`),
  - bias energy component scaling over \(T_o\),
  - shaping vs column resistance and temperature.

See `SigmaDeltaModulator::GetReadPathEnergy` / `SigmaDeltaModulator::GetInputPathEnergy` in `SigmaDeltaModulator.cpp` for the exact expressions.

---

## 3. Where it is implemented (code map)

| Layer | File | Role |
|--------|------|------|
| **Configuration** | `NeuroSIM/Param.cpp` | Defines `cimInterfaceMode` (`BASELINE_ADC` vs `SIGMA_DELTA_STREAM`), default ΣΔ knobs (`eascimNaturalFreqFc`, `eascimObservationPeriodTo`, `eascimCintFemtoFarad`, `eascimIrefNanoAmp`, `eascimSigmaDeltaVdd`, …), and **env overrides** for sweeps. |
| **Subarray readout** | `NeuroSIM/SubArray.cpp` | Chooses SAR ADC vs ΣΔ modulator based on `cimInterfaceMode`. In ΣΔ mode, it routes area/latency/energy through `sigmaDeltaModulator` (output encoder) and can also instantiate `sigmaDeltaModulatorRow` (input encoder) for specific cell types. |
| **PE (processing unit)** | `NeuroSIM/ProcessingUnit.cpp` | In ΣΔ mode, it **does not perform** the baseline PE-level digital adder-tree/buffer/bus accounting in the same way as ADC mode (so these contributions are gated off in area/performance buckets). |
| **Tile** | `NeuroSIM/Tile.cpp` | In ΣΔ mode, the tile-level digital accumulation/latency/energy path is simplified/bypassed for the main datapath in favor of the subarray ΣΔ timing. |
| **Chip** | `NeuroSIM/Chip.cpp` | If ΣΔ pool+activation is enabled, the model uses `AnalogStreamNonlinearity.*` to compute max-pool comparator budgeting and analog-style ReLU/sigmoid PPA. Otherwise it falls back to digital pool/activation PPA. |
| **Python → C++** | `Inference_pytorch/inference.py` (and hooks) | `--inference 1` triggers hooks → logs/traces → calls the NeuroSim C++ backend (`main`). |
| **Sweeps / logs** | `Inference_pytorch/scripts/run_cs6501_project*.sh` | Runs baseline and ΣΔ grids using environment variables (`NS_CIM_INTERFACE_MODE`, `EASCIM_FC_HZ`, `EASCIM_TO_S`, etc.). |
| **Summaries** | `Inference_pytorch/scripts/parse_project_results.py` | Parses per-run logs → `summary.csv` (accuracy, latency/clk, energy, TOPS/W, FPS, area). |

---

## 4. How a run flows end-to-end

1. **Environment**: activate the intended Conda env (e.g., `conda activate neurosim`) so PyTorch/CUDA and the NeuroSim build/runtime are consistent.
2. **Python inference**: `inference.py` loads CIFAR-10 and the selected DNN (e.g., VGG8 or DenseNet40), runs a test pass, and uses hooks to export layer traces.
3. **C++ backend**: NeuroSim `main` reads `Param` + layer traces and evaluates hierarchical PPA (floorplan, subarray/module PPA, chip-level latency/energy/area).
4. **ΣΔ vs baseline selection**: determined by `NS_CIM_INTERFACE_MODE`:
   - `NS_CIM_INTERFACE_MODE=0` → baseline ADC path
   - `NS_CIM_INTERFACE_MODE=1` → ΣΔ stream path + gated/bypassed digital hierarchy

---

## 5. Parameters you sweep (experiment side)

Typical sweep grid (as used in project logs):

- `NS_CIM_INTERFACE_MODE`: `0` baseline, `1` ΣΔ.
- `EASCIM_FC_HZ`: natural stream frequency \(f_c\) (e.g., `50e6`, `100e6`, `200e6`).
- `EASCIM_TO_S`: observation window \(T_o\) (e.g., `1e-8`, `3e-8`, `6e-8`, …).
- `EASCIM_SD_VDD_V`: ΣΔ analog supply voltage (default is typically `0.4` V in code).
- Optional ΣΔ knobs:
  - `EASCIM_CINT_FF` (integration capacitor in fF),
  - `EASCIM_IREF_NA` (reference current in nA),
  - `EASCIM_ROW_SDM_ACCOUNT` (row ΣΔ area/energy separate accounting),
  - `EASCIM_ANALOG_POOL_ACT` (pool/activation PPA style under ΣΔ),
  - `EASCIM_ANALOG_E_PJ` (override energy per compare event for analog stream pool/activation PPA),
  - `EASCIM_ANALOG_AREA_FACTOR` (override area factor for analog stream pool/activation PPA).
- `subArray` / `parallelRead` and DNN quantization knobs:
  - constrained by the wrapper logic (e.g., `parallelRead == subArray` when `cellBit > 1`).

Results are aggregated with `parse_project_results.py`.

---

## 6. Relation to the papers (intent)

The implementation is intended to reflect **stream-based, time-encoded readout** and **reduced reliance on conventional digital accumulation / buffering / activation** along the PE–tile–chip path—consistent with the intent of EAS-CiM / EASI-CiM / ΣΔ-stream modeling.

However, NeuroSim remains an **architectural PPA estimator**, not a cycle-accurate mixed-signal simulator. Therefore:

1. The code exposes paper-aligned architectural knobs:
   - observation window \(T_o\) (`eascimObservationPeriodTo`),
   - natural stream frequency \(f_c\) (`eascimNaturalFreqFc`),
   - integration capacitor \(C_\mathrm{int}\) (`eascimCintFemtoFarad`),
   - feedback reference current \(I_\mathrm{ref}\) (`eascimIrefNanoAmp`),
   - ΣΔ analog supply \(V_\mathrm{dd}\) (`eascimSigmaDeltaVdd`).
2. Energy/latency are computed using first-order algebraic expressions inside `SigmaDeltaModulator` and (optionally) inside `AnalogStreamNonlinearity`. These are used for **consistent architectural accounting** inside NeuroSim’s existing PPA framework.
3. In ΣΔ mode, the simulator uses a **clock-decoupling assumption** for FPS/TOPS in this fork:
   - the “global clock period” is derived from a synchronous-equivalent latency that does not fully stretch with the output ΣΔ observation window.
   - Energy and end-to-end read accounting still include ΣΔ contributions.
   
So the reported PPA comparisons should be interpreted as **consistent first-order architectural estimates** aligned with the stream-based evaluation intent, not as a full SPICE validation.

---

## 7. Analog stream max-pool and activation (optional PPA)

When **`cimInterfaceMode == SIGMA_DELTA_STREAM`** and **`eascimAnalogPoolActivate`** is true (default under ΣΔ unless overridden), `Chip.cpp` routes max-pooling area and chip-level ReLU/sigmoid PPA through:

- `AnalogStreamNonlinearity` (`NeuroSIM/AnalogStreamNonlinearity.h/.cpp`)

This is an **architectural** PPA model. It budgets:
- comparator event counts tied to pool window size, MPU-style tree depth, and the observation/aggregation latency derived from \(T_o\) and \(f_c\),
- dynamic energy per compare event using `eascimAnalogEnergyPerComparePJ` (overridable via `EASCIM_ANALOG_E_PJ`),
- area using a comparator-per-\(F^2\) style budget scaled by `eascimAnalogCompAreaBeta` and an area factor.

### Environment variables
- `EASCIM_ANALOG_POOL_ACT`:
  - `0` = use digital `MaxPooling` / activation blocks under ΣΔ
  - `1` = use analog stream comparator PPA model
- `EASCIM_ANALOG_E_PJ`:
  - optional override for energy per compare event (pJ)
  
`main` prints the ΣΔ configuration under the CiM interface banner so logs show whether the analog stream pool/activation path was enabled.

---

## References

- R. Sreekumar *et al.*, **“EASI-CiM: Event-driven Asynchronous Stream-based Image Classifier with Compute-in-Memory Kernels”**, *IEEE ISQED*, 2024.
- R. Sreekumar *et al.*, **“EAS-CiM 2.0: Event-driven Asynchronous Stream-based Compute-in-Memory Kernels with Scalable Precision”**, *IEEE ISCAS*, 2025.
- P.-Y. Chen, X. Peng, S. Yu, **“NeuroSim: A Circuit-Level Macro Model for Benchmarking Neuro-Inspired Architectures in Online Learning”**, *IEEE TCAD*, 2018.
- DNN+NeuroSim V1.4: *User manual / repository for framework details*.
