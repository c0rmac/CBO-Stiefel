// filename: objectives/gep_objective.h

#ifndef GEP_OBJECTIVE_H
#define GEP_OBJECTIVE_H

#include "../base_objective.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver
#include <stdexcept>
#include <numeric>
#include <vector>
#include <algorithm>

class GEPObjective : public BaseObjective {
public:
    // --- Constructor Updated ---
    // Added optional seed parameter
    GEPObjective(int d, unsigned int seed = 54321) // Default seed
        : BaseObjective(d, d), seed_(seed) // Store the seed
    {
        // Generate A using the fixed seed
        generate_random_symmetric_matrix();
        true_minimum_ = calculate_true_minimum();
    }

    // --- Core Interface Implementation (calculate_cost, get_minimizer, get_true_minimum remain the same) ---
    double calculate_cost(const MatrixNxK& X) const override {
        if (X.rows() != n_ || X.cols() != k_) {
             throw std::runtime_error("GEPObjective: Input matrix X has incorrect dimensions.");
        }
        double trace_val = (X.transpose() * A_ * X).trace();
        return -trace_val;
    }

    const MatrixNxK& get_minimizer() const override {
        if (X_placeholder_.rows() == 0) {
             X_placeholder_ = MatrixNxK::Zero(n_, k_);
        }
        return X_placeholder_;
    }

    double get_true_minimum() const override {
        return true_minimum_;
    }

private:
    MatrixNxN A_; // d x d Symmetric matrix
    double true_minimum_;
    unsigned int seed_; // Store the seed
    inline static MatrixNxK X_placeholder_;

    // --- Matrix Generation (Updated with Local Seeding) ---
    void generate_random_symmetric_matrix() {
        // --- Use a local, seeded generator ---
        std::mt19937 local_gen(seed_);
        // Use Eigen's normal distribution random generator if available,
        // otherwise use std::normal_distribution
        // Eigen::Rand::PCLDU<double> normal_dist; // Example alternative if needed
        std::normal_distribution<> dist(0.0, 1.0); // Standard normal

        // Generate Z using the local generator
        MatrixNxN Z = MatrixNxN::NullaryExpr(n_, n_, [&]() { return dist(local_gen); });
        // --- End Local Seeding ---

        A_ = (Z + Z.transpose()) / 2.0; // Ensure symmetry
    }

    // --- Analytical Minimum Calculation (Remains the same) ---
    double calculate_true_minimum() const {
        Eigen::SelfAdjointEigenSolver<MatrixNxN> es(A_);
        VectorN eigenvalues = es.eigenvalues();
        double min_gep_value = -eigenvalues.sum();
        return min_gep_value;
    }
};

#endif // GEP_OBJECTIVE_H