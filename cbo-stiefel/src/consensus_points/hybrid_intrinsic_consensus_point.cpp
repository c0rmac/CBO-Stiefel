// filename: consensus_points/hybrid_intrinsic_consensus_point.cpp

#include "hybrid_intrinsic_consensus_point.h"
#include <numeric>

MatrixNxK HybridIntrinsicConsensusPoint::calculate(const std::vector<MatrixNxK>& particles) {
    if (particles.empty() || particles.size() != N_) {
        return MatrixNxK::Zero(n_, k_);
    }

    // 1. Calculate stable weights
    VectorN normalized_weights = calculate_weights_stable(particles);

    // 2. Compute the extrinsic weighted sum (Initial Guess calculation)
    MatrixNxK extrinsic_mean = MatrixNxK::Zero(n_, k_);
    bool calculation_possible = false;
    for (int i = 0; i < N_; ++i) {
        if (normalized_weights[i] > 1e-12 && particles[i].allFinite()) {
            extrinsic_mean += normalized_weights[i] * particles[i];
            calculation_possible = true;
        }
    }
    // Handle fallback if extrinsic mean calculation failed
    if (!calculation_possible || !extrinsic_mean.allFinite()) {
        // Recalculate as simple mean of finite particles
        extrinsic_mean.setZero();
        int valid_count = 0;
        for (const auto& p : particles) {
            if (p.allFinite()) { extrinsic_mean += p; valid_count++; }
        }
        if (valid_count > 0) { extrinsic_mean /= valid_count; }
        if (!extrinsic_mean.allFinite()) return MatrixNxK::Zero(n_, k_); // Final failsafe
    }

    // 3. Project extrinsic mean onto the manifold to get Y_0
    MatrixNxK current_mean = BaseCBOSolver::project_manifold_eigen(extrinsic_mean);

    // 4. Perform Riemannian Gradient Descent steps (if max_gradient_steps > 0)
    for (int step = 0; step < max_gradient_steps_; ++step) {
        MatrixNxK gradient_sum = MatrixNxK::Zero(n_, k_); // Accumulator for gradient
        bool gradient_valid = false;

        // Calculate gradient: V = sum_i w_i * Log_map(current_mean, particle_i)
        for (int i = 0; i < N_; ++i) {
            if (normalized_weights[i] > 1e-12 && particles[i].allFinite()) {
                MatrixNxK log_map_vec = log_map(current_mean, particles[i]); // Use placeholder Log
                if (log_map_vec.allFinite()) {
                    gradient_sum += normalized_weights[i] * log_map_vec;
                    gradient_valid = true;
                }
            }
        }

        // If gradient calculation failed, stop iterating
        if (!gradient_valid || !gradient_sum.allFinite()) {
            // std::cerr << "Warning: Gradient calculation failed in Hybrid Intrinsic step " << step << std::endl;
            break; // Exit the gradient descent loop
        }

        // Update: Y_{k+1} = Exp_map(Y_k, -step_size * Gradient)
        MatrixNxK update_tangent = -step_size_ * gradient_sum;
        current_mean = exp_map(current_mean, update_tangent); // Use placeholder Exp

        // Ensure the result is still finite after Exp map
        if (!current_mean.allFinite()){
             // std::cerr << "Warning: Exp map resulted in non-finite mean in Hybrid Intrinsic step " << step << std::endl;
             // Revert to projection of extrinsic mean as failsafe
             current_mean = BaseCBOSolver::project_manifold_eigen(extrinsic_mean);
             break;
        }
    }

    // Return the final computed mean (after 0 or more GD steps)
    return current_mean;
}