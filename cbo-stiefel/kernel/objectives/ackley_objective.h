// filename: kernel/objectives/ackley_objective.h

#ifndef ACKLEY_OBJECTIVE_H
#define ACKLEY_OBJECTIVE_H

#include "base_objective.h" // Inherit from the new base class
#include <cmath>
#include <limits>
#include <Eigen/Core>

// Use the existing type definitions
// using MatrixNxK = Eigen::MatrixXd; // Inherited from base_objective.h

class AckleyObjective : public BaseObjective {
public:
    // --- Constants from Paper ---
    static constexpr double DEFAULT_A = 20.0;
    static constexpr double DEFAULT_B = 0.2;
    static constexpr double DEFAULT_C = 3.0;

    // Use the base class constructor to initialize n, k, and nk
    AckleyObjective(int n, int k, double a = DEFAULT_A, double b = DEFAULT_B, double c = DEFAULT_C)
        : BaseObjective(n, k), a_(a), b_(b), c_(c)
    {
        X_star_ = create_minimizer(n, k);
    }

    // --- Core Interface Implementation (Override Base Methods) ---

    // 1. Cost Calculation (Implements the pure virtual method)
    double calculate_cost(const MatrixNxK& X) const override {
        MatrixNxK diff = X - X_star_;

        // 1. Term 1: Exponential of Euclidean distance
        // This calculation matches the structure where:
        // -a_ is the outer multiplier (A=20.0 in Python script)
        // -b_ is the scale factor before the sqrt (a=0.2 in Python script)
        // -c_ is the factor inside the sqrt (b=3.0 in Python script)
        double diff_norm_sq = diff.squaredNorm();
        double term1_exp_inner = std::sqrt(c_ * c_ * diff_norm_sq / nk_);
        double term1_exp = -b_ * term1_exp_inner;
        double term1 = -a_ * std::exp(term1_exp);

        // 2. Term 2: Exponential of cosine average
        // Uses c_ for the cosine scaling, matching the standard formula.
        double cos_sum = (2.0 * M_PI * c_ * diff.array()).cos().sum();
        double term2_inner = cos_sum / nk_;
        double term2 = -std::exp(term2_inner);

        // Final value: f(X) = Term1 + Term2 + e + A
        return term1 + term2 + std::exp(1.0) + a_;
    }

    // 2. Minimizer Getter (Implements the pure virtual method)
    const MatrixNxK& get_minimizer() const override {
        return X_star_;
    }

    // 3. True Minimum Getter (Implements the pure virtual method)
    double get_true_minimum() const override {
        return TRUE_MINIMUM; // Use constant defined in base class context
    }

private:
    double a_;
    double b_;
    double c_;
    MatrixNxK X_star_;

    // --- Helper to Create X* (I_{n,k}) ---
    static MatrixNxK create_minimizer(int n, int k) {
        MatrixNxK X_star = MatrixNxK::Zero(n, k);
        // Set the top k x k block to Identity
        X_star.topLeftCorner(k, k).setIdentity();
        return X_star;
    }
    // Using TRUE_MINIMUM from BaseObjective declaration
    static constexpr double TRUE_MINIMUM = 0.0;
};

#endif // ACKLEY_OBJECTIVE_H