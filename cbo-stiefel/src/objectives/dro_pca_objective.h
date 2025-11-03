// filename: dro_pca_objective.h

#ifndef DRO_PCA_OBJECTIVE_H
#define DRO_PCA_OBJECTIVE_H

#include "base_objective.h"
#include <Eigen/Core>
#include <cmath>       // For std::sqrt, std::max
#include <stdexcept>   // For std::invalid_argument, std::runtime_error
#include <limits>      // For std::numeric_limits

// Type alias for n x n matrices (d x d in the paper)
using MatrixNxN = Eigen::MatrixXd;

/**
 * @brief Implements the Distributionally Robust PCA (DRO-PCA) objective function.
 *
 * This class implements the equivalent reformulation (P_m) [cite: 52, 53] from the paper
 * "Enhancing Distributional Robustness in Principal Component Analysis by Wasserstein Distances"
 * (arXiv:2503.02494v2).
 *
 * The objective function is defined by (4.1)  as f(X) = u(X) + s(X) + w(X), where:
 * 1. u(X) = tr((I_n - XX^T) * Sigma_0) [cite: 270]
 * 2. s(X) = gamma * ||X||_1 (l1-norm regularizer, as used in experiments) [cite: 569, 570]
 * 3. w(X) = 2 * rho * ||(I_n - XX^T) * Sigma_0^(1/2)||_F [cite: 269]
 *
 * A key simplification from (3.7) [cite: 236] and [cite: 251] allows us to rewrite w(X) using
 * the trace:
 * w(X) = 2 * rho * sqrt(tr((I_n - XX^T) * Sigma_0))
 *
 * So the final function implemented is:
 * f(X) = tr(C(X)) + gamma * ||X||_1 + 2 * rho * sqrt(tr(C(X)))
 * where C(X) = (I_n - XX^T) * Sigma_0
 */
class DroPcaObjective : public BaseObjective {
public:
    /**
     * @brief Constructs the DRO-PCA objective.
     *
     * @param n Dimension of the space (d in the paper)[cite: 17].
     * @param k Number of principal components (r in the paper)[cite: 20].
     * @param nominal_covariance The n x n nominal covariance matrix (Sigma_0)[cite: 54].
     * @param rho The radius of the Wasserstein ball (rho >= 0)[cite: 45].
     * @param gamma The l1-norm regularization parameter (gamma >= 0)[cite: 568].
     */
    DroPcaObjective(int n, int k, const MatrixNxN& nominal_covariance, double rho, double gamma)
        : BaseObjective(n, k),
          Sigma0_(nominal_covariance), // Make a copy for safety
          rho_(rho),
          gamma_(gamma)
    {
        if (nominal_covariance.rows() != n || nominal_covariance.cols() != n) {
            throw std::invalid_argument("Nominal covariance matrix must be n x n.");
        }
        if (rho < 0.0) {
            throw std::invalid_argument("Wasserstein radius rho must be non-negative.");
        }
        if (gamma < 0.0) {
            throw std::invalid_argument("Regularization parameter gamma must be non-negative.");
        }
    }

    virtual ~DroPcaObjective() = default;

    /**
     * @brief Calculates the cost of the DRO-PCA objective function f(X).
     *
     * @param X The n x k matrix of principal components (X in the paper).
     * @return The objective function value f(X).
     */
    virtual double calculate_cost(const MatrixNxK& X) const override {
        if (X.rows() != n_ || X.cols() != k_) {
            throw std::runtime_error("Input matrix X has incorrect dimensions.");
        }

        // 1. Calculate the projection matrix P = (I_n - XX^T)
        // X is n x k (d x r)
        MatrixNxN XXt = X * X.transpose(); // (n x k) * (k x n) -> (n x n)
        MatrixNxN P = MatrixNxN::Identity(n_, n_) - XXt;

        // 2. Calculate the core trace term: tr((I_n - XX^T) * Sigma_0)
        // This is tr(P * Sigma_0).
        double trace_C = (P * Sigma0_).trace();

        // The trace term represents a variance [cite: 18] and is the argument
        // for a square root[cite: 251]. It must be non-negative.
        // We use std::max for numerical stability.
        double trace_term = std::max(0.0, trace_C);

        // 3. Calculate u(X) = tr((I_n - XX^T) * Sigma_0) [cite: 270]
        double u_X = trace_term;

        // 4. Calculate s(X) = gamma * ||X||_1 [cite: 569, 570]
        double s_X = gamma_ * X.cwiseAbs().sum();

        // 5. Calculate w(X) = 2 * rho * sqrt(tr((I_n - XX^T) * Sigma_0)) [cite: 269, 251]
        double w_X = 2.0 * rho_ * std::sqrt(trace_term);

        // Total cost f(X) = u(X) + s(X) + w(X)
        return u_X + s_X + w_X;
    }

    /**
     * @brief Returns a placeholder for the true minimizer.
     * As the true minimizer is unknown, this returns a default-constructed
     * (0x0) static matrix.
     */
    virtual const MatrixNxK& get_minimizer() const override {
        return dummy_minimizer_;
    }

    /**
     * @brief Returns the known true minimum value.
     * This is set to 0.0, as the true minimum is not known.
     */
    virtual double get_true_minimum() const override {
        return 0.0;
    }

protected:
    MatrixNxN Sigma0_;  // The n x n nominal covariance matrix (Sigma_0)
    double rho_;        // Wasserstein radius
    double gamma_;      // l1 regularization parameter

    // Static placeholder for the unknown minimizer.
    // C++17 allows inline initialization of static members.
    inline static MatrixNxK dummy_minimizer_ = MatrixNxK(0, 0);
};

#endif // DRO_PCA_OBJECTIVE_H