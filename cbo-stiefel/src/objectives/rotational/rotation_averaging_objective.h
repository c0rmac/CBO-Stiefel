// filename: objectives/rotation_averaging_objective.h

#ifndef ROTATION_AVERAGING_OBJECTIVE_H
#define ROTATION_AVERAGING_OBJECTIVE_H

#include "../base_objective.h"
#include <vector>
#include <Eigen/Core>
#include <stdexcept>
#include <limits> // For infinity/NaN

// Assumes optimization is over SO(d), which means n=k=d.

class RotationAveragingObjective : public BaseObjective {
public:
    // --- Constructor ---
    // Takes the dimension 'd' and the measurement data.
    RotationAveragingObjective(int d,
                               const std::vector<MatrixNxK>& measurements_Z, // List of Z_j matrices (d x d)
                               const std::vector<MatrixNxK>& references_X_star) // List of X_j* matrices (d x d)
        : BaseObjective(d, d), // n = k = d for SO(d)
          measurements_Z_(measurements_Z),
          references_X_star_(references_X_star)
    {
        if (measurements_Z_.empty() || references_X_star_.empty()) {
            throw std::invalid_argument("Rotation Averaging requires at least one measurement pair.");
        }
        if (measurements_Z_.size() != references_X_star_.size()) {
            throw std::invalid_argument("Number of measurements Z_j must match number of references X_j*.");
        }
        // Basic dimension check on the first pair (assuming consistency)
        if (measurements_Z_[0].rows() != d || measurements_Z_[0].cols() != d ||
            references_X_star_[0].rows() != d || references_X_star_[0].cols() != d) {
            throw std::invalid_argument("Measurement and reference matrices must be d x d.");
        }
        // Note: We don't explicitly enforce SO(d) constraint here,
        // the solver's retraction/projection handles the manifold.
        // However, reference matrices X_j* should ideally be in SO(d).
    }

    // --- Core Interface Implementation ---

    double calculate_cost(const MatrixNxK& X) const override {
        // Cost: f(X) = sum_{j} || Z_j - X.T @ X_j* ||_F^2

        double total_cost = 0.0;
        MatrixNxK Xt = X.transpose(); // Precompute transpose

        for (size_t j = 0; j < measurements_Z_.size(); ++j) {
            // Calculate the residual for pair j: Z_j - X.T @ X_j*
            MatrixNxK residual_j = measurements_Z_[j] - (Xt * references_X_star_[j]);

            // Add the squared Frobenius norm to the total cost
            total_cost += residual_j.squaredNorm();
        }

        return total_cost;
    }

    // The true minimizer X* is generally unknown without solving the problem.
    // Return a placeholder.
    const MatrixNxK& get_minimizer() const override {
        // Create a static placeholder if not already done
        if (X_placeholder_.rows() == 0) { // Lazy initialization
             X_placeholder_ = MatrixNxK::Zero(n_, k_);
        }
        return X_placeholder_;
    }

    // The true minimum value depends on noise in Z_j.
    double get_true_minimum() const override {
        // If measurements were perfect (Z_j = (X_true)^T X_j*), the minimum would be 0.
        // In practice, it will be > 0 due to noise. Return NaN as unknown.
        return std::numeric_limits<double>::quiet_NaN();
    }

private:
    std::vector<MatrixNxK> measurements_Z_;
    std::vector<MatrixNxK> references_X_star_;
    // Static placeholder for get_minimizer
    inline static MatrixNxK X_placeholder_;
};

#endif // ROTATION_AVERAGING_OBJECTIVE_H