// filename: consensus_points/hybrid_intrinsic_consensus_point.h

#ifndef HYBRID_INTRINSIC_CONSENSUS_POINT_H
#define HYBRID_INTRINSIC_CONSENSUS_POINT_H

#include "base_consensus_point.h"
#include "../solvers/base_solver.h" // For projection
#include <vector>
#include <Eigen/Core>
#include <string>

class HybridIntrinsicConsensusPoint : public BaseConsensusPoint {
public:
    HybridIntrinsicConsensusPoint(int n, int k, int N_particles, double beta_val,
                                  BaseObjective* objective_ptr,
                                  int max_gradient_steps = 1, // Configurable steps
                                  double step_size = 0.1)     // Step size for gradient descent
        : BaseConsensusPoint(n, k, N_particles, beta_val, objective_ptr),
          max_gradient_steps_(max_gradient_steps),
          step_size_(step_size)
    {
        if (max_gradient_steps_ < 0) {
            throw std::invalid_argument("max_gradient_steps must be non-negative.");
        }
    }

    // Override the calculate method
    MatrixNxK calculate(const std::vector<MatrixNxK>& particles) override;

    std::string get_name() const override { return "Hybrid Intrinsic (Projected + GD)"; }

private:
    int max_gradient_steps_;
    double step_size_;

    // --- Placeholder Riemannian Geometry Functions ---
    // !!! IMPORTANT: Replace these with actual implementations for V(k,n) or SO(d) !!!
    MatrixNxK log_map(const MatrixNxK& base, const MatrixNxK& target) const {
        // Placeholder: Approximates Log map by projecting the difference vector.
        // This is NOT geometrically correct but allows compilation.
        return BaseCBOSolver::project_tangent_eigen(target - base, base);
    }

    MatrixNxK exp_map(const MatrixNxK& base, const MatrixNxK& tangent_vec) const {
        // Placeholder: Approximates Exp map by adding tangent vector and re-projecting.
        // This is NOT geometrically correct but allows compilation.
        return BaseCBOSolver::project_manifold_eigen(base + tangent_vec);
    }
    // --- End Placeholder Functions ---
};

#endif // HYBRID_INTRINSIC_CONSENSUS_POINT_H