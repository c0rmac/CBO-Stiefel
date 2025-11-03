// filename: objectives/opp_objective.h

#ifndef OPP_OBJECTIVE_H
#define OPP_OBJECTIVE_H

#include "../base_objective.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU> // Required for determinant
#include <stdexcept>
#include <limits>
#include <random> // For seeding

class OPPObjective : public BaseObjective {
public:
    // --- Constructor Updated ---
    // Added 'enforce_SOd' boolean flag
    OPPObjective(int d, unsigned int seed = 67890, bool enforce_SOd = true)
        : BaseObjective(d, d), // n = k = d
          seed_(seed),
          enforce_SOd_(enforce_SOd) // Store the constraint flag
    {
        generate_random_matrices();
        calculate_analytical_solution(); // This function now respects the flag
    }

    // --- Core Interface Implementation (calculate_cost, get_minimizer, get_true_minimum) ---
    double calculate_cost(const MatrixNxK& X) const override {
        // Cost: f(X) = ||A @ X - B||_F^2
        if (X.rows() != n_ || X.cols() != k_) {
            throw std::runtime_error("OPPObjective: Input matrix X has incorrect dimensions.");
        }
        MatrixNxN residual = (A_ * X) - B_;
        return residual.squaredNorm();
    }

    const MatrixNxK& get_minimizer() const override {
        return X_star_analytical_;
    }

    double get_true_minimum() const override {
        return true_minimum_;
    }

private:
    MatrixNxN A_; // d x d matrix
    MatrixNxN B_; // d x d matrix
    MatrixNxN X_star_analytical_; // The d x d orthogonal/special orthogonal matrix X*
    double true_minimum_;
    unsigned int seed_;
    bool enforce_SOd_; // Flag to enforce determinant +1

    // --- Matrix Generation ---
    void generate_random_matrices() {
        std::mt19937 local_gen(seed_);
        std::normal_distribution<> dist(0.0, 1.0);
        A_ = MatrixNxN::NullaryExpr(n_, n_, [&]() { return dist(local_gen); });
        B_ = MatrixNxN::NullaryExpr(n_, n_, [&]() { return dist(local_gen); });
    }

    // --- Calculate Analytical Solution (Updated for SO(d)) ---
    void calculate_analytical_solution() {
        // For min ||AX - B||_F^2, X in O(d) or SO(d)
        // Solution involves SVD of A^T B.
        MatrixNxN AtB = A_.transpose() * B_;
        Eigen::JacobiSVD<MatrixNxN> svd(AtB, Eigen::ComputeFullU | Eigen::ComputeFullV);
        MatrixNxN U = svd.matrixU();
        MatrixNxN V = svd.matrixV();
        VectorN S_diag = svd.singularValues(); // Sigma values (vector)

        // Determinant of the O(d) minimizer (V * U.transpose())
        double det_VU_T = (V * U.transpose()).determinant();

        if (enforce_SOd_ && det_VU_T < 0.0) {
            // --- Enforce SO(d) (det = +1) ---
            // We must use the modified solution X* = V * D * U.T
            // where D = diag(1, 1, ..., -1)
            MatrixNxN D = MatrixNxN::Identity(n_, n_);
            D(n_ - 1, n_ - 1) = -1.0; // Flip the sign of the last singular value's contribution

            X_star_analytical_ = V * D * U.transpose();

            // The true minimum for SO(d) is Tr(A^T A + B^T B) - 2 * Tr(Sigma * D)
            // Tr(Sigma * D) = (S_1 + ... + S_{d-1} - S_d)
            double trace_Sigma_D = S_diag.head(n_ - 1).sum() - S_diag(n_ - 1);
            true_minimum_ = A_.squaredNorm() + B_.squaredNorm() - 2.0 * trace_Sigma_D;

        } else {
            // --- Standard O(d) (det = +/-1) ---
            X_star_analytical_ = V * U.transpose();

            // The true minimum for O(d) is Tr(A^T A + B^T B) - 2 * Tr(Sigma)
            double trace_Sigma = S_diag.sum();
            true_minimum_ = A_.squaredNorm() + B_.squaredNorm() - 2.0 * trace_Sigma;
        }
    }
};

#endif // OPP_OBJECTIVE_H