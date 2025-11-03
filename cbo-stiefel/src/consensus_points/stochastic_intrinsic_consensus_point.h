// filename: consensus_points/stochastic_intrinsic_consensus_point.h

#ifndef STOCHASTIC_INTRINSIC_CONSENSUS_POINT_H
#define STOCHASTIC_INTRINSIC_CONSENSUS_POINT_H

#include "base_consensus_point.h"
#include "../solvers/base_solver.h" // For projection
#include <vector>
#include <Eigen/Core>
#include <string>
#include <iostream>

class StochasticIntrinsicConsensusPoint : public BaseConsensusPoint {
public:
    StochasticIntrinsicConsensusPoint(int n, int k, int N_particles, double beta_val,
                                      BaseObjective* objective_ptr,
                                      double step_size = 0.1) // Step size for SGD update
        : BaseConsensusPoint(n, k, N_particles, beta_val, objective_ptr),
          step_size_(step_size),
          is_initialized_(false) // Flag to handle first step
    {
         // Initialize prev_consensus_point_ to something reasonable (e.g., zero or random projection)
         MatrixNxK Z_rand = MatrixNxK::Random(n, k);
         prev_consensus_point_ = BaseCBOSolver::project_manifold_eigen(Z_rand);
    }

    // Override the calculate method
    MatrixNxK calculate(const std::vector<MatrixNxK>& particles) override;

    std::string get_name() const override { return "Stochastic Intrinsic (Online)"; }

private:
    double step_size_;
    MatrixNxK prev_consensus_point_; // Store the mean from the previous step
    bool is_initialized_;

    // --- Placeholder Riemannian Geometry Functions ---
    // !!! IMPORTANT: Replace these with actual implementations !!!
    MatrixNxK log_map(const MatrixNxK& base, const MatrixNxK& target) const {
        return BaseCBOSolver::project_tangent_eigen(target - base, base); // Placeholder
    }

    MatrixNxK exp_map(const MatrixNxK& base, const MatrixNxK& tangent_vec) const {
        return BaseCBOSolver::project_manifold_eigen(base + tangent_vec); // Placeholder
    }
    // --- End Placeholder Functions ---
};

#endif // STOCHASTIC_INTRINSIC_CONSENSUS_POINT_H