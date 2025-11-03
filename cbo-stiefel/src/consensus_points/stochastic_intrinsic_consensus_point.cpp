// filename: consensus_points/stochastic_intrinsic_consensus_point.cpp

#include "stochastic_intrinsic_consensus_point.h"
#include <numeric>

MatrixNxK StochasticIntrinsicConsensusPoint::calculate(const std::vector<MatrixNxK>& particles) {
    if (particles.empty() || particles.size() != N_) {
        return MatrixNxK::Zero(n_, k_);
    }

    // 1. Calculate stable weights using current particle positions
    VectorN normalized_weights = calculate_weights_stable(particles);

    // --- Use previous mean as the base point for Log/Exp maps ---
    MatrixNxK base_point;
    if (!is_initialized_) {
        // For the very first step, calculate extrinsic mean and project it as starting point
        MatrixNxK extrinsic_first = MatrixNxK::Zero(n_, k_);
        int valid_count = 0;
         for (int i = 0; i < N_; ++i) {
            if (normalized_weights[i] > 1e-12 && particles[i].allFinite()) {
                extrinsic_first += normalized_weights[i] * particles[i];
                valid_count++;
            }
         }
         if (valid_count > 0) {
             base_point = BaseCBOSolver::project_manifold_eigen(extrinsic_first);
         } else {
             // If calculation failed, use the random initial point
             base_point = prev_consensus_point_;
         }
        is_initialized_ = true; // Mark as initialized
    } else {
        base_point = prev_consensus_point_; // Use mean from previous step
    }
    // Ensure base point is valid
    if (!base_point.allFinite()) base_point = MatrixNxK::Identity(n_, k_); // Final failsafe


    // 2. Calculate the "stochastic" gradient at the base point
    // V = sum_i w_i * Log_map(base_point, particle_i)
    MatrixNxK gradient_sum = MatrixNxK::Zero(n_, k_);
    bool gradient_valid = false;
    for (int i = 0; i < N_; ++i) {
        if (normalized_weights[i] > 1e-12 && particles[i].allFinite()) {
            MatrixNxK log_map_vec = log_map(base_point, particles[i]); // Use placeholder Log
            if (log_map_vec.allFinite()) {
                gradient_sum += normalized_weights[i] * log_map_vec;
                gradient_valid = true;
            }
        }
    }

    MatrixNxK current_mean; // The new mean to calculate

    // 3. Perform one step update if gradient is valid
    if (gradient_valid && gradient_sum.allFinite()) {
        MatrixNxK update_tangent = -step_size_ * gradient_sum;
        current_mean = exp_map(base_point, update_tangent); // Use placeholder Exp

        // Ensure result is finite
        if (!current_mean.allFinite()){
             // std::cerr << "Warning: Exp map resulted in non-finite mean in Stochastic Intrinsic." << std::endl;
             current_mean = base_point; // Failsafe: keep previous mean
        }
    } else {
        // If gradient calculation failed, keep the previous mean
        current_mean = base_point;
    }

    // 4. Store the newly computed mean for the next step
    prev_consensus_point_ = current_mean;

    return current_mean;
}