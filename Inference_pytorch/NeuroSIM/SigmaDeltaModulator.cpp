/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
*
* Model aligned with EAS-CiM 2.0 (ISCAS 2025): Fig. 4 ΣΔ modulator uses I_in vs. 1-bit
* feedback, switched I_ref onto C_int, DLS inverter with hysteresis; precision scales with
* T_o and f_c (Fig. 2). Circuits in the paper are simulated in 65 nm;MiM C_int bank and I_ref
* ranges follow the manuscript. MiM area scaling uses Param (eascimAreaMimPerFfM2, eascimAreaOverheadFactor);
* tune DLS_INV_EQ / energy coefficients in code or Param against Virtuoso if you need exact netlist agreement.
********************************************************************************/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "SigmaDeltaModulator.h"

using namespace std;

extern Param *param;

/* Paper-reported ranges (EAS-CiM 2.0, Section III, Fig. 4). */
/* Paper-reported programmable knob ranges (Fig. 4 / Section III). */
static const double EASCIM_CINT_MIN_FF = 40.0;
static const double EASCIM_CINT_MAX_FF = 160.0;
static const double EASCIM_IREF_MIN_NA = 10.0;
static const double EASCIM_IREF_MAX_NA = 35.0;
/* Reference technology for analog block scaling in the paper. */
static const double EASCIM_REF_FEATURE_M = 65e-9;
/* DLS inverter, hysteresis, integration switches: inverter-equivalent count (order-of-magnitude). */
static const double DLS_INV_EQ = 14.0;
static const double IREF_MIRROR_INV_EQ = 10.0;

SigmaDeltaModulator::SigmaDeltaModulator(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell)
	: inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void SigmaDeltaModulator::Initialize(int _numCol, double _naturalFreqFc, double _observationPeriodTo,
		double _cIntFemtoFarad, double _iRefNanoAmp, int _numReadCellPerOperationNeuro, SigmaDeltaRole _role) {
	if (initialized) {
		cout << "[SigmaDeltaModulator] Warning: Already initialized!" << endl;
	} else {
		numCol = _numCol;
		naturalFreqFc = max(1.0, _naturalFreqFc);
		observationPeriodTo = max(1e-15, _observationPeriodTo);
		cIntFemtoFarad = min(max(_cIntFemtoFarad, EASCIM_CINT_MIN_FF), EASCIM_CINT_MAX_FF);
		iRefNanoAmp = min(max(_iRefNanoAmp, EASCIM_IREF_MIN_NA), EASCIM_IREF_MAX_NA);
		numReadCellPerOperationNeuro = _numReadCellPerOperationNeuro;
		role = _role;

		widthNmos = MIN_NMOS_SIZE * tech.featureSize;
		widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

		initialized = true;
	}
}

double SigmaDeltaModulator::GetPulseCountInObservationWindow() const {
	if (!initialized)
		return 1.0;
	return max(1.0, naturalFreqFc * observationPeriodTo);
}

void SigmaDeltaModulator::CalculateUnitArea() {
	if (!initialized) {
		cout << "[SigmaDeltaModulator] Error: Require initialization first!" << endl;
		return;
	}
	double hNmos, wNmos, hPmos, wPmos;
	CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNmos, &wNmos);
	CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hPmos, &wPmos);
	double invCell = hNmos * wNmos + hPmos * wPmos;
	double nodeScale = tech.featureSize / EASCIM_REF_FEATURE_M;
	nodeScale = max(0.5, min(nodeScale, 2.0));

	double areaTransistor = invCell * (DLS_INV_EQ + IREF_MIRROR_INV_EQ) * nodeScale;
	double areaMim = param->eascimAreaMimPerFfM2 * cIntFemtoFarad * param->eascimAreaOverheadFactor;

	areaUnit = areaTransistor + areaMim;
}

void SigmaDeltaModulator::CalculateArea(double heightArray, double widthArray, AreaModify _option) {
	if (!initialized) {
		cout << "[SigmaDeltaModulator] Error: Require initialization first!" << endl;
		return;
	}

	area = 0;
	height = 0;
	width = 0;

	if (widthArray && _option == NONE) {
		area = areaUnit * numCol;
		width = widthArray;
		height = area / widthArray;
	} else if (heightArray && _option == NONE) {
		area = areaUnit * numCol;
		height = heightArray;
		width = area / height;
	} else {
		cout << "[SigmaDeltaModulator] Error: No width or height assigned for the sigma-delta peripheral" << endl;
		exit(-1);
	}

	newHeight = heightArray;
	newWidth = widthArray;
	switch (_option) {
		case MAGIC:
			MagicLayout();
			break;
		case OVERRIDE:
			OverrideLayout();
			break;
		default:
			break;
	}
}

void SigmaDeltaModulator::CalculateLatency(double numRead) {
	if (!initialized) {
		cout << "[SigmaDeltaModulator] Error: Require initialization first!" << endl;
		return;
	}
	readLatency = 0;
	/* EAS-CiM timing: observation/service window is T_o. */
	readLatency += observationPeriodTo * numRead;
}

void SigmaDeltaModulator::CalculatePower(const vector<double> &columnResistance, double numRead) {
	if (!initialized) {
		cout << "[SigmaDeltaModulator] Error: Require initialization first!" << endl;
		return;
	}
	leakage = 0;
	readDynamicEnergy = 0;
	if (role == SigmaDeltaRole::OUTPUT_ENCODER) {
		for (size_t i = 0; i < columnResistance.size(); i++) {
			readDynamicEnergy += GetReadPathEnergy(columnResistance[i]);
		}
	} else {
		/* Input-side encoder cost is not a function of output column resistance. */
		readDynamicEnergy = numCol * GetInputPathEnergy();
	}
	readDynamicEnergy *= numRead;
}

void SigmaDeltaModulator::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

double SigmaDeltaModulator::GetReadPathEnergy(double columnRes) {
	const double Varray = param->readVoltage;
	const double Vsd = max(0.0, param->eascimSigmaDeltaVdd);
	double columnResScaled = columnRes * 0.5 / max(Varray, 1e-12);

	double cInt = cIntFemtoFarad * 1e-15;
	double iRef = iRefNanoAmp * 1e-9;
	double pulseN = GetPulseCountInObservationWindow();

	/* Switching on C_int ∝ pulse count; bias path ∝ T_o (paper: higher f_c → higher dynamic power). */
	double eSwitch = pulseN * 0.5 * cInt * Vsd * Vsd * param->eascimEnergySwitchFactor;
	double eBias = observationPeriodTo * iRef * Vsd * param->eascimEnergyBiasFactor;

	double colFactor = 1.0;
	if ((double)1 / columnResScaled == 0) {
		colFactor = 0.01;
	} else if (columnResScaled == 0) {
		colFactor = 0;
	} else {
		colFactor = 0.55 + 0.45 * exp(-1.85 * log10(max(columnResScaled, 1.0)));
	}

	double tempFactor = (1 + 1.3e-3 * (param->temp - 300));
	return (eSwitch + eBias) * colFactor * tempFactor;
}

double SigmaDeltaModulator::GetInputPathEnergy() const {
	const double Vsd = max(0.0, param->eascimSigmaDeltaVdd);
	double cInt = cIntFemtoFarad * 1e-15;
	double iRef = iRefNanoAmp * 1e-9;
	double pulseN = GetPulseCountInObservationWindow();
	double eSwitch = pulseN * 0.5 * cInt * Vsd * Vsd * param->eascimEnergySwitchFactor;
	double eBias = observationPeriodTo * iRef * Vsd * param->eascimEnergyBiasFactor;
	double tempFactor = (1 + 1.3e-3 * (param->temp - 300));
	return (eSwitch + eBias) * tempFactor;
}
