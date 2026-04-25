/*******************************************************************************
 * AnalogStreamNonlinearity — see AnalogStreamNonlinearity.h
 *******************************************************************************/

#include <algorithm>
#include <cmath>
#include "AnalogStreamNonlinearity.h"
#include "Param.h"
#include "MaxPooling.h"

using namespace std;

int AnalogStreamNonlinearity::comparatorsPerMpu(int window) {
	int n = window;
	int m = n % 2;
	int numComparator = 0;
	while (n != 0) {
		int add = n / 2;
		numComparator += add;
		n /= 2;
	}
	numComparator += m;
	return max(1, numComparator);
}

double AnalogStreamNonlinearity::observationLatencyS(const Param& p) {
	double to_s = p.eascimObservationPeriodTo;
	double settle_s = 0.0;
	if (p.eascimNaturalFreqFc > 0.0)
		settle_s = p.eascimSettleCycles / p.eascimNaturalFreqFc;
	return to_s + settle_s;
}

double AnalogStreamNonlinearity::maxPoolChipAreaUm2(const Param& p, const MaxPooling& pool) {
	if (!pool.initialized)
		return 0.0;
	int compPerMpu = comparatorsPerMpu(pool.window);
	long long totalComp = (long long)compPerMpu * (long long)max(1, pool.numMaxPooling);
	double f_um = p.featuresize / 1e-6;
	return (double)totalComp * p.eascimAnalogCompAreaBeta * f_um * f_um;
}

double AnalogStreamNonlinearity::reluChipAreaUm2(const Param& p, int numUnit, int numBit) {
	(void)numBit;
	long long nC = (long long)max(1, numUnit);
	double f_um = p.featuresize / 1e-6;
	return (double)nC * p.eascimAnalogCompAreaBeta * f_um * f_um * 0.35;
}

void AnalogStreamNonlinearity::maxPoolAnalogPerf(const Param& p, const MaxPooling& pool, double numRead,
	double* latency_s, double* dynamic_energy_j) {
	if (!pool.initialized) {
		*latency_s = 0.0;
		*dynamic_energy_j = 0.0;
		return;
	}
	const double per_event_s = observationLatencyS(p);
	*latency_s = per_event_s * max(1.0, numRead);
	long long compEvents = (long long)ceil(numRead) * (long long)max(1, pool.numMaxPooling)
		* (long long)comparatorsPerMpu(pool.window);
	const double e_per_compare_j = p.eascimAnalogEnergyPerComparePJ * 1e-12;  // pJ -> J
	*dynamic_energy_j = (double)compEvents * e_per_compare_j * p.eascimAnalogStreamEnergyFactor;
}

void AnalogStreamNonlinearity::reluAnalogPerf(const Param& p, double numRead, int numUnit,
	double* latency_s, double* dynamic_energy_j) {
	const double per_event_s = observationLatencyS(p);
	*latency_s = per_event_s * max(1.0, numRead);
	long long compEvents = (long long)ceil(numRead) * (long long)max(1, numUnit);
	const double e_per_compare_j = p.eascimAnalogEnergyPerComparePJ * 1e-12;  // pJ -> J
	*dynamic_energy_j = (double)compEvents * e_per_compare_j * p.eascimAnalogStreamEnergyFactor;
}
