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
* Extension: SigmaDeltaModulator — asynchronous ΣΔ / ASC stream generator as in EAS-CiM 2.0
* (R. Sreekumar et al., "EAS-CiM 2.0: Event-driven Asynchronous Stream-based Compute-in-Memory
* Kernels with Scalable Precision," IEEE ISCAS 2025). Encoding: duty cycle δ and frequency f
* vs. normalized input p follow the modulator temporal transfer (see paper Eq. (1)(2)).
* Hardware knobs modeled: integration capacitor C_int (MiM bank), reference current I_ref,
* natural frequency f_c, observation window T_o (latency vs. precision).
********************************************************************************/

#ifndef SIGMADELTAMODULATOR_H_
#define SIGMADELTAMODULATOR_H_

#include <vector>
#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "FunctionUnit.h"

using namespace std;

class SigmaDeltaModulator: public FunctionUnit {
public:
	enum class SigmaDeltaRole {
		INPUT_ENCODER = 0,	// WL / activation stream generator
		OUTPUT_ENCODER = 1	// SL / readout stream re-encoder
	};

	SigmaDeltaModulator(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell);
	virtual ~SigmaDeltaModulator() {}
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;

	/* Functions */
	void PrintProperty(const char* str);
	/* EAS-CiM 2.0: f_c [Hz], T_o [s], C_int [fF] (paper: 40–160 fF), I_ref [nA] (10–35 nA). */
	void Initialize(int _numCol, double _naturalFreqFc, double _observationPeriodTo,
			double _cIntFemtoFarad, double _iRefNanoAmp, int _numReadCellPerOperationNeuro,
			SigmaDeltaRole _role = SigmaDeltaRole::OUTPUT_ENCODER);
	void CalculateUnitArea();
	void CalculateArea(double heightArray, double widthArray, AreaModify _option);
	void CalculateLatency(double numRead);
	void CalculatePower(const vector<double> &columnResistance, double numRead);
	double GetReadPathEnergy(double columnRes);
	double GetInputPathEnergy() const;

	/* Expected pulse count in one T_o window; used internally for energy (≈ f_c · T_o). */
	double GetPulseCountInObservationWindow() const;

	/* Properties */
	bool initialized;
	int numCol;
	SigmaDeltaRole role;
	/* f_c: natural frequency of the modulator at zero input (paper). */
	double naturalFreqFc;
	/* T_o: finite observation / aggregation period — primary latency vs. precision knob. */
	double observationPeriodTo;
	/* Integrator MiM capacitor bank (paper Fig. 4: programmable 40 fF … 160 fF). */
	double cIntFemtoFarad;
	/* Switched reference current (paper: 10 nA … 35 nA cascade mirror). */
	double iRefNanoAmp;
	double widthNmos, widthPmos;
	double areaUnit;
	int numReadCellPerOperationNeuro;
};

#endif /* SIGMADELTAMODULATOR_H_ */
