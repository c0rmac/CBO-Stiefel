// filename: objectives/qap_objective.h

#ifndef QAP_OBJECTIVE_H
#define QAP_OBJECTIVE_H

#include "base_objective.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues> // Required to compute true minimum using eigenvalues
#include <Eigen/SVD>         // Useful for complex matrix ops/normalization
#include <stdexcept>
#include <algorithm>
#include <numeric>

// The QAP is defined on the orthogonal group O(n), which is V(n, n) in the paper's context.

class QAPObjective : public BaseObjective {
public:
    // --- Constants ---
    // The true minimum is calculated analytically, not a fixed constant.

    QAPObjective(int n, int k, unsigned int seed = 12345)
        : BaseObjective(n, k), seed_(seed)
    {
        if (n != k) {
            // The QAP relaxation is typically defined over the orthogonal group O(n),
            // where X is n x n (i.e., V(n, n)).
            throw std::invalid_argument("QAP is typically defined over V(n,n). n must equal k.");
        }

        // Generate the random symmetric matrices A and B
        generate_symmetric_matrices();

        // Calculate the true minimum based on the generated A and B
        true_minimum_ = calculate_true_minimum();
    }

    // --- Core Interface Implementation (Override Base Methods) ---

    double calculate_cost(const MatrixNxK& X) const override {
        // Cost: f(X) = tr(A @ X @ B @ X.T)

        // Note: X must be n x n here (as n=k)
        MatrixNxK term1 = A_ * X;
        MatrixNxK term2 = term1 * B_;
        MatrixNxK term3 = term2 * X.transpose();

        // Return the trace of the final matrix product
        return term3.trace();
    }

    // Since the minimization argument X* is complex (reordering of eigenvectors),
    // we return a zero matrix as a placeholder for the argument, focusing on the value.
    const MatrixNxK& get_minimizer() const override {
        static const MatrixNxK X_placeholder = MatrixNxK::Zero(n_, k_);
        return X_placeholder;
    }

    double get_true_minimum() const override {
        return true_minimum_;
    }

private:
    MatrixNxN A_; // n x n Symmetric
    MatrixNxN B_; // n x n Symmetric
    double true_minimum_;
    unsigned int seed_;

    // --- Matrix Generation Logic ---
    void generate_symmetric_matrices() {
        // --- Use a local, seeded generator ---
        // This ensures matrix generation is deterministic for a given seed,
        // without affecting the global random state used by the CBO solver.
        std::mt19937 local_gen(seed_);
        std::uniform_real_distribution<> dist(0.0, 1.0); // U(0, 1)

        // Generate Z1 and Z2 using the local generator
        MatrixNxN Z1 = MatrixNxN::NullaryExpr(n_, n_, [&]() { return dist(local_gen); });
        MatrixNxN Z2 = MatrixNxN::NullaryExpr(n_, n_, [&]() { return dist(local_gen); });
        // --- End Local Seeding ---

        A_ = (Z1 + Z1.transpose()) / 2.0;
        B_ = (Z2 + Z2.transpose()) / 2.0;
    }

    // --- Analytical Minimum Calculation ---
    double calculate_true_minimum() const {
        // Minimum QAP value = sum_{i=1}^n (lambda_i * mu_{n-i})
        // where lambda_i and mu_i are sorted ascendingly [cite: 1468]

        // 1. Compute Eigenvalues for A and B
        Eigen::SelfAdjointEigenSolver<MatrixNxN> es_a(A_);
        Eigen::SelfAdjointEigenSolver<MatrixNxN> es_b(B_);

        VectorN lambda = es_a.eigenvalues();
        VectorN mu = es_b.eigenvalues();

        // Eigen::SelfAdjointEigenSolver sorts eigenvalues ascendingly by default.

        double min_qap_value = 0.0;
        for (int i = 0; i < n_; ++i) {
            // Pair the i-th smallest eigenvalue of A (lambda[i])
            // with the (n-i-1)-th smallest (i.e., i-th largest) eigenvalue of B (mu[n-i-1])
            min_qap_value += lambda[i] * mu[n_ - 1 - i];
        }

        return min_qap_value;
    }
};

#endif // QAP_OBJECTIVE_H