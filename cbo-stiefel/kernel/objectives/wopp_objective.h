// filename: kernel/objectives/wopp_objective.h

#ifndef WOPP_OBJECTIVE_H
#define WOPP_OBJECTIVE_H

#include "base_objective.h" // Inherit from the abstract base class
#include <Eigen/Core>
#include <Eigen/QR>         // Required for QR factorization (to generate orthogonal matrices)
#include <stdexcept>
#include <iostream>

// The base class defines MatrixNxK = Eigen::MatrixXd

class WOPPObjective : public BaseObjective {
public:
    // --- Constants ---
    static constexpr double TRUE_MINIMUM = 0.0; // Guaranteed by construction

    WOPPObjective(int n, int k)
        : BaseObjective(n, k)
    {
        // Generate the random coefficient matrices A, B, C such that the problem is well-defined
        generate_random_matrices();
    }

    // --- Core Interface Implementation (Override Base Methods) ---

    double calculate_cost(const MatrixNxK& X) const override {
        // X must be n x k. The solver ensures this.

        // Calculate Residual: R = A @ X @ C - B
        MatrixNxK residual = (A_ * X * C_) - B_;

        // Calculate Cost: f(X) = 0.5 * ||Residual||_F^2
        // Eigen's squaredNorm() computes the square of the Frobenius norm.
        double f_X = 0.5 * residual.squaredNorm();

        return f_X;
    }

    const MatrixNxK& get_minimizer() const override {
        // This problem guarantees min f(X)=0 at X=X_star_rand.
        return X_star_rand_;
    }

    double get_true_minimum() const override {
        return TRUE_MINIMUM;
    }

private:
    MatrixNxN A_; // n x n matrix
    MatrixNxK B_; // n x k matrix
    MatrixKxK C_; // k x k matrix

    MatrixNxK X_star_rand_; // The known minimizer on V(n,k)

    // --- Matrix Generation Logic ---

    void generate_random_matrices() {
        // The generation logic follows the paper's specification [cite: 1496-1499]:

        // 1. Generate A = P @ S @ R.T (n x n)
        // P and R are random orthogonal, S is diagonal U(0, 2)
        Eigen::HouseholderQR<MatrixNxN> qr_p(MatrixNxN::Random(n_, n_));
        MatrixNxN P = qr_p.householderQ();

        Eigen::HouseholderQR<MatrixNxN> qr_r(MatrixNxN::Random(n_, n_));
        MatrixNxN R = qr_r.householderQ();

        VectorN s_diag = VectorN::Random(n_).array().abs() * 1.0; // U(0, 2) approximation
        // Ensure values are within the range U(0, 2)
        s_diag = s_diag.cwiseMin(2.0);
        s_diag = s_diag.cwiseMax(1e-6); // Avoid zero/negative

        MatrixNxN S = s_diag.asDiagonal();
        A_ = P * S * R.transpose();

        // 2. Generate C = Q @ T @ Q.T (k x k)
        // Q is random orthogonal, T is diagonal U(0, 10)
        Eigen::HouseholderQR<MatrixKxK> qr_q(MatrixKxK::Random(k_, k_));
        MatrixKxK Q = qr_q.householderQ();

        VectorN t_diag(k_);
        t_diag.setRandom();
        t_diag = (t_diag.array().abs() * 5.0) + 5.0; // U(0, 10) approximation
        t_diag = t_diag.cwiseMin(10.0).cwiseMax(1e-6);

        MatrixKxK T = t_diag.asDiagonal();
        C_ = Q * T * Q.transpose();

        // 3. Generate a random true minimizer X* on V(k, n)
        MatrixNxK Z_rand = MatrixNxK::Random(n_, k_);
        Eigen::HouseholderQR<MatrixNxK> qr_x_star(Z_rand);

        // --- CRITICAL FIX ---
        // 3a. Force computation of the full Q matrix into a temporary variable.
        MatrixNxN Q_full = qr_x_star.householderQ();

        // 3b. X_star is the first k columns of the orthonormal Q matrix.
        // Q_full has size n x n, we only need the first k columns for X* (n x k).
        X_star_rand_ = Q_full.leftCols(k_);
        // --- End CRITICAL FIX ---

        // 4. Generate B = A @ X* @ C, guaranteeing f(X*) = 0
        B_ = A_ * X_star_rand_ * C_;
    }
};

#endif // WOPP_OBJECTIVE_H