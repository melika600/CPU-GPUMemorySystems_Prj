# Sigma-delta (ΣΔ) stream interface in NeuroSim (`Inference_pytorch`)

---

## Big picture

In a normal **analog-to-digital** path you might sample a voltage once with a multi-bit **SAR ADC**. In the **ΣΔ stream** view used here, the array still produces an analog result (for example **column current** after a multiply–accumulate along a bit line), but the **readout interface** is treated like a **1-bit stream over time**: the hardware keeps integrating, comparing, and feeding back until the **average** of that fast 1-bit stream represents the value you care about.

Three plain steps:

1. **Integrate** — Charge builds on an **integration capacitor** (**C_int**, in femtofarads). The net current into that node (input minus feedback) moves the voltage on **C_int**.
2. **Decide** — A **comparator** (in the paper, a hysteresis inverter) flips when the integrator crosses a band. That decision is your **1-bit output** for a slice of time.
3. **Feedback** — A **reference current** (**I_ref**, in nanoamps) is switched on or off so the integrator is **pushed back** toward balance. That closing of the loop is what makes the **average duty cycle** of the bit stream track the input.

You do not get a multi-bit code in one shot. You get **accuracy from time**: if you watch the stream for a longer **observation window** (**T_o**, in seconds), the **average** of the bits is a better estimate of the analog value. If the internal oscillator runs faster (**f_c**, in Hz), you can get **more transitions inside the same T_o**, which usually improves effective resolution but costs **more switching energy** on **C_int**.

**Important:** NeuroSim is **not** running SPICE. It is a **PPA** (power, performance, area) **architectural** model: the code uses **C_int**, **I_ref**, **T_o**, **f_c**, and the ΣΔ supply **V_dd** to produce **reasonable latency, energy, and area** that you can compare across designs and technology knobs.

---

## 1. Where the mode is selected

Everything starts in **`Param`**.

`Param::cimInterfaceMode` picks which readout story the subarray uses:

| Mode | What it means |
|------|----------------|
| `BASELINE_ADC` | **Classic NeuroSim.** Use the usual SAR / MLSA style paths controlled by the existing SAR and MLSA flags. |
| `SIGMA_DELTA_STREAM` | **ΣΔ stream interface.** On the **column / sense side**, the simulator uses the **ΣΔ output encoder** (`sigmaDeltaModulator`) for the same “slot” where SAR latency, energy, and area used to plug in (via shared macros in `SubArray.cpp`). |

**Row side vs column side — stated simply:**

- **Column (output) ΣΔ** — Models the **readout stream** after the array. This is always part of the ΣΔ story when `SIGMA_DELTA_STREAM` is on.
- **Row (input / WL) ΣΔ** — For **RRAM** and **FeFET** cell types, the code can also model a **row-side** modulator (`sigmaDeltaModulatorRow`) for **timing**. Whether its **area and energy** are **booked as a separate block** is controlled by **`eascimRowSigmaDeltaSeparateAccounting`**: if `false`, you still may get **timing** from the row modulator, but you do **not** split out its area/energy as its own line item (similar in spirit to not breaking out every input DAC detail in baseline flows).

---

## 2. The six knobs (names in code vs meaning on paper)

These names appear in **`Param`** and in **`SigmaDeltaModulator::Initialize`**.

| In code (`Param` / modulator) | Symbol (paper / intuition) | Units | What it does in simple words |
|-------------------------------|-----------------------------|-------|-------------------------------|
| `eascimObservationPeriodTo` | **T_o** | seconds | **How long you “watch” the stream”** to form one digital observation. **Longer T_o** → usually **better effective precision**, **more latency** per read in the model. |
| `eascimNaturalFreqFc` | **f_c** | Hz | **How fast the modulator can toggle internally** when it is active. **Higher f_c** → **more pulses inside the same T_o** → tends to **better resolution** and **higher dynamic energy** (more charging/discharging of **C_int**). |
| `eascimCintFemtoFarad` | **C_int** | fF (femtofarads) | **Integration capacitor bank** size. **Larger C_int** → same charge change produces **smaller voltage step** → often **finer** quantization of the integrator node, but **more charge** moved per switch → affects **energy** and **MiM area** in the model. |
| `eascimIrefNanoAmp` | **I_ref** | nA (nanoamps) | **Strength of the feedback reference current.** Sets how aggressively the loop pushes the integrator back; scales **bias-related energy** over **T_o** in the energy model. |
| `eascimSigmaDeltaVdd` | **V_dd** (ΣΔ block) | volts | **Supply voltage used for ΣΔ switching-energy math** (for example **0.4 V** in a low-voltage analog style). It is **not** automatically every other block’s Vdd; it is the **analog ΣΔ periphery** supply you assign for energy. |
| `eascimRowSigmaDeltaSeparateAccounting` | (no single symbol) | bool | **`true`** — count **row ΣΔ** area/energy as its **own** bucket. **`false`** — **do not** split row ΣΔ area/energy out; **timing** for row ΣΔ may still apply where the code path includes it. |

**Clamp in code:** **`C_int`** and **`I_ref`** are **clamped** to the paper’s reported ranges inside **`SigmaDeltaModulator::Initialize`** so stray config values do not blow up the model.

**Roles inside the C++ class:**

- **`OUTPUT_ENCODER`** — Column / readout side (`sigmaDeltaModulator`).
- **`INPUT_ENCODER`** — Row / word-line side (`sigmaDeltaModulatorRow`) when that path is built.

---

## 3. How data moves through this repo (Python → CSV → C++)

```
  models/*.py, modules/*.py, utee/hook.py
              |
              v
     layer_record_*  (CSV: weights and activations per layer)
              |
              v
     NeuroSIM: main.cpp reads Param, builds hierarchy
              |
              v
          SubArray  ------>  sigmaDeltaModulator (OUTPUT_ENCODER)
              |
              +-- (RRAM / FeFET) -->  sigmaDeltaModulatorRow (INPUT_ENCODER)
                                       timing optional in PPA (see flag above)
```

1. **Python** runs the network and **hooks** export tensors to **CSV** files under folders like `layer_record_VGG8/`.
2. **`main`** (C++) reads **`Param`** (defaults in **`Param.cpp`**), reads the layer traces, and runs **latency, area, and energy** calculators for tiles and subarrays.
3. **`SubArray`** chooses SAR vs ΣΔ using macros such as **`NS_USE_SIGMA_DELTA`**. In ΣΔ mode, **`sigmaDeltaModulator`** is a normal **`FunctionUnit`**: it gets **`Initialize`**, **`CalculateLatency`**, **`CalculatePower`**, **`CalculateArea`**, just like other blocks.

---

## 4. Latency: what the simulator actually adds

For the **output** ΣΔ block, **`SigmaDeltaModulator::CalculateLatency(numRead)`** does one simple thing:

**ΣΔ output latency = T_o × numRead**

In code that is:

**`readLatency` (for that block) += `observationPeriodTo` * `numRead`**

So:

- **`numRead`** is how NeuroSim counts read operations in that path.
- Every counted read pays **one full observation window** **T_o** in the model.

There is **no separate “settle” term** added in that function today. If you need settling to be explicit, you fold it into your choice of **T_o** / **f_c** / experiment setup rather than a second line in that formula.

**Row ΣΔ:** When the row modulator is present, it uses the **same latency pattern**. The **subarray’s total `readLatency`** can therefore include **both** output and row pieces, depending on the **`NS_SUBROW_SDM_RL`** path in **`SubArray.cpp`**.

---

## 5. `readLatency` vs `readLatencySync` (why two numbers?)

**Problem in one sentence:** If you put the **full** ΣΔ observation time **T_o × numRead** into the same bucket that sets the **global chip clock**, a long **T_o** makes the whole design look **artificially slow** (low FPS), even when you want to explore an **asynchronous** ΣΔ periphery that does **not** force the whole digital chip to run that slowly.

**What NeuroSim does:**

| Variable | Meaning |
|----------|---------|
| **`readLatency`** | **Full** modeled read path time, including **output ΣΔ** observation time and **row ΣΔ** timing when that path is active. Use this when you care about **true end-to-end read time** with ΣΔ on the critical path. |
| **`readLatencySync`** | Starts from **`readLatency`**, then **subtracts only the output ΣΔ block’s** **`sigmaDeltaModulator.readLatency`** when **`SIGMA_DELTA_STREAM`** is on. **Row ΣΔ is not subtracted** here. |

**`ProcessingUnit.cpp`** uses **`readLatencySync`** when it updates **`clkPeriod`**. So:

- **FPS / global clock** track a **“synchronous-style”** view that **does not stretch** with the **output** ΣΔ window.
- **Energy** and the **full** **`readLatency`** path can still **include** ΣΔ so you do not lose physical cost in the model.

**Rule of thumb:**

- Ask “how long does a read take **including** watching the ΣΔ stream?” → look at **`readLatency`**.
- Ask “how fast does the **rest of the chip** tick in this experiment?” → look at **`clkPeriod` / FPS** derived from **`readLatencySync`**.

---

## 6. Energy and area 

- **Area** — A **transistor budget** (scaled from a **65 nm** style reference in the code comments) plus a **metal–insulator–metal (MiM)** capacitor area that **grows with C_int**, using **`eascimAreaMimPerFfM2`** and **`eascimAreaOverheadFactor`**.
- **Dynamic energy (output path)** — For each column, the model adds:
  - **Switching energy** on **C_int** that scales with roughly **how many effective pulses** you get in a window (the code ties this to **f_c × T_o** through **`GetPulseCountInObservationWindow`**), scaled by **`eascimEnergySwitchFactor`**, and using **V_dd** for the **0.5·C·V²** style term.
  - **Bias energy** over **T_o** using **I_ref** and **V_dd**, scaled by **`eascimEnergyBiasFactor`**.
  - Extra **column-resistance** and **temperature** shaping as implemented in **`GetReadPathEnergy`**.

For the exact formulas, open **`SigmaDeltaModulator.cpp`** — functions **`GetReadPathEnergy`** and **`GetInputPathEnergy`**.

---

## 7. Quick parameter checklist (`Param`)

| Field | Plain words |
|-------|-------------|
| `cimInterfaceMode` | **`BASELINE_ADC`** = classic SAR/MLSA style. **`SIGMA_DELTA_STREAM`** = use ΣΔ accounting on the column side. |
| `eascimNaturalFreqFc` | Internal **f_c** [Hz]. |
| `eascimObservationPeriodTo` | **T_o** [s]. |
| `eascimCintFemtoFarad` | **C_int** [fF], clamped in **`Initialize`**. |
| `eascimIrefNanoAmp` | **I_ref** [nA], clamped in **`Initialize`**. |
| `eascimSigmaDeltaVdd` | **V_dd** for the ΣΔ block in **energy** math (example: **0.4** V). |
| `eascimRowSigmaDeltaSeparateAccounting` | Split **row ΣΔ** **area/energy** out (`true`) or keep it merged (`false`). |

---

## 8. Example logs shipped with this tree

Under **`Inference_pytorch/`**:

- **`logrun/VGGcifar10Compare.log`** — Compare-style CIFAR-10 / VGG NeuroSim runs (good for **before/after** or **delta percent** tables).
- **`logrun2/cifar10_vgg8_sigmadelta_paperDefaults_Vdd0p4_rowModeledNotCounted_clkDecoupled.log`** — VGG8 with **paper-style defaults**, **V_dd = 0.4 V** for ΣΔ energy, **row ΣΔ modeled but not separately counted** in PPA, and **clock derived from `readLatencySync`** (ΣΔ observation **decoupled** from **`clkPeriod`**).

---

## References

- R. Sreekumar *et al.*, “EAS-CiM 2.0: Event-driven Asynchronous Stream-based Compute-in-Memory Kernels with Scalable Precision,” *IEEE ISCAS*, 2025.
- NeuroSim: P.-Y. Chen, X. Peng, S. Yu, Arizona State University (see headers under **`NeuroSIM/`**).
