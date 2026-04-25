/*******************************************************************************
 * AnalogStreamNonlinearity — lightweight PPA for ΣΔ-aligned analog-ish
 * max-pool and activation (integrate + compare / event-style), without
 * widening the baseline multi-bit digital datapath.
 *
 * Used when Param::cimInterfaceMode == SIGMA_DELTA_STREAM and
 * Param::eascimAnalogPoolActivate is true.
 *******************************************************************************/

#ifndef ANALOG_STREAM_NONLINEARITY_H_
#define ANALOG_STREAM_NONLINEARITY_H_

class Param;
class MaxPooling;

struct AnalogStreamPpa {
	double area_um2;
	// IMPORTANT: keep units consistent with NeuroSim core:
	// - latency in seconds
	// - energy in Joules
	double latency_s;
	double dynamic_energy_j;
};

class AnalogStreamNonlinearity {
public:
	/** Comparator count in one max-pool MPU (same reduction tree as MaxPooling::Initialize). */
	static int comparatorsPerMpu(int window);

	/** Integrate-and-compare style latency floor from T_o and f_c (seconds). */
	static double observationLatencyS(const Param& p);

	/** Chip-level max-pool area (µm²): MPUs × comparators × β·F². */
	static double maxPoolChipAreaUm2(const Param& p, const MaxPooling& pool);

	/** Chip-level ReLU analog area (µm²): comparators ~ numUnit (integrate–compare per unit). */
	static double reluChipAreaUm2(const Param& p, int numUnit, int numBit);

	/** Dynamic energy (J) and latency (s) for one max-pool evaluation batch (numRead as in MaxPooling::CalculatePower). */
	static void maxPoolAnalogPerf(const Param& p, const MaxPooling& pool, double numRead,
		double* latency_s, double* dynamic_energy_j);

	/** Same for chip ReLU (numRead as in BitShifter::CalculatePower / GreLu path), returns (s, J). */
	static void reluAnalogPerf(const Param& p, double numRead, int numUnit,
		double* latency_s, double* dynamic_energy_j);
};

#endif /* ANALOG_STREAM_NONLINEARITY_H_ */
