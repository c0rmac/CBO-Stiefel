// filename: consensus_points/base_consensus_point.h

#ifndef BASE_CONSENSUS_POINT_H
#define BASE_CONSENSUS_POINT_H

#include "../objectives/base_objective.h"
#include "../solvers/base_solver.h" // Include base solver for projection methods
#include <vector>
#include <Eigen/Core>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>

using MatrixNxK = Eigen::MatrixXd;
using VectorN = Eigen::VectorXd;

class BaseConsensusPoint {
public:
    BaseConsensusPoint(int n, int k, int N_particles, double beta_val, BaseObjective* objective_ptr)
        : n_(n), k_(k), N_(N_particles), beta_(beta_val), objective_ptr_(objective_ptr)
    {
        if (!objective_ptr_) {
            throw std::invalid_argument("Objective pointer cannot be null for consensus calculation.");
        }
    }

    virtual ~BaseConsensusPoint() = default;

    // --- Abstract Method: Calculate the consensus point ---
    virtual MatrixNxK calculate(const std::vector<MatrixNxK>& particles) = 0;

    // --- Optional: Method to get the name ---
    virtual std::string get_name() const = 0;

protected:
    // --- Helper to calculate stable Boltzmann weights ---
    VectorN calculate_weights_stable(const std::vector<MatrixNxK>& particles) const {
        // ... (implementation remains the same as before) ...
        VectorN costs(N_);
        bool any_finite_cost = false;
        double min_cost = std::numeric_limits<double>::infinity();

        for (int i = 0; i < N_; ++i) {
            costs[i] = objective_ptr_->calculate_cost(particles[i]);
            if (std::isfinite(costs[i])) {
                any_finite_cost = true;
                min_cost = std::min(min_cost, costs[i]);
            }
        }

        VectorN normalized_weights = VectorN::Zero(N_);
        if (!any_finite_cost) {
             if (N_ > 0) normalized_weights.fill(1.0 / N_);
             return normalized_weights;
        }

        VectorN weights_unnormalized = (-beta_ * (costs.array() - min_cost)).exp();
        for(int i=0; i < N_; ++i) {
            if(!std::isfinite(costs[i])) weights_unnormalized[i] = 0.0;
        }

        double sum_weights = weights_unnormalized.sum();

        if (sum_weights < 1e-100 || !std::isfinite(sum_weights)) {
            int valid_count = 0;
            for(int i=0; i < N_; ++i) { if(std::isfinite(costs[i])) valid_count++; }
            if (valid_count > 0) {
                 double uniform_weight = 1.0 / valid_count;
                 for(int i=0; i < N_; ++i) {
                     if(std::isfinite(costs[i])) normalized_weights[i] = uniform_weight;
                 }
            } else if (N_ > 0) {
                 normalized_weights.fill(1.0 / N_);
            }
        } else {
            normalized_weights = weights_unnormalized / sum_weights;
        }
        return normalized_weights;
    }


    // --- Placeholder Riemannian Geometry Functions ---
    // !!! IMPORTANT: Replace these with actual implementations for V(k,n) or SO(d) !!!
    // Moved here from derived classes.
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

    // Member variables
    int n_;
    int k_;
    int N_;
    double beta_;
    BaseObjective* objective_ptr_;
};

#endif // BASE_CONSENSUS_POINT_H