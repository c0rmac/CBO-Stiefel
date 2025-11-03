// filename: consensus_points/extrinsic_consensus_point.cpp

#include "extrinsic_consensus_point.h"
#include <numeric> // For fallback mean calculation if needed

MatrixNxK ExtrinsicConsensusPoint::calculate(const std::vector<MatrixNxK>& particles) {
    if (particles.empty() || particles.size() != N_) {
         // Handle error or return zero matrix
         return MatrixNxK::Zero(n_, k_);
    }

    // 1. Calculate stable weights using the base class method
    VectorN normalized_weights = calculate_weights_stable(particles);

    // 2. Compute the weighted sum
    MatrixNxK consensus_point = MatrixNxK::Zero(n_, k_);
    bool calculation_possible = false;
    for (int i = 0; i < N_; ++i) {
        // Only include particles with positive weight and valid data
        if (normalized_weights[i] > 1e-12 && particles[i].allFinite()) {
            consensus_point += normalized_weights[i] * particles[i];
            calculation_possible = true;
        }
    }

    // 3. Final check / Fallback if all weights were zero or particles invalid
    if (!calculation_possible || !consensus_point.allFinite()) {
        // Fallback: Calculate simple mean of all finite particles
        consensus_point.setZero();
        int valid_count = 0;
        for (const auto& p : particles) {
            if (p.allFinite()) {
                consensus_point += p;
                valid_count++;
            }
        }
        return (valid_count > 0) ? consensus_point / valid_count : MatrixNxK::Zero(n_, k_);
    }

    return consensus_point;
}