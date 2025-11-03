// filename: kim_solver.cpp

#include "solvers/kim_solver.h" // Include the corresponding header
#include <cmath>
#include <numeric>
#include <limits> // For numeric_limits
#include <Eigen/QR>

// --- Constructor Definition (Matches Header) ---
KimCBOSolver::KimCBOSolver(int n, int k,
                           BaseObjective* objective_ptr, // Changed parameter
                           int N_particles, double beta_val, double lambda_val, double sigma_val,
                           const std::string& backend,
                           bool enforce_SOd)
    // Pass objective_ptr to the BaseCBOSolver constructor
    : BaseCBOSolver(n, k, objective_ptr, N_particles, beta_val, backend, enforce_SOd),
      lambda_(lambda_val), sigma_(sigma_val) {}

// --- Core SDE Step Logic ---
void KimCBOSolver::run_single_step(double h) {
    // calculate_consensus_stable() now uses objective_ptr_ internally
    MatrixNxK M = calculate_consensus_stable();
    std::vector<MatrixNxK> new_particles(N_);
    double sigma_sq = sigma_ * sigma_;
    double sqrt_h = std::sqrt(h);
    MatrixNxN Id_n = MatrixNxN::Identity(n_, n_); // n x n Identity matrix

    for (int i = 0; i < N_; ++i) {
        const MatrixNxK& Xi = particles_[i];
        double dist_XM_sq = (Xi - M).squaredNorm(); // Squared Frobenius norm
        double dist_XM = std::sqrt(dist_XM_sq);

        // --- 1. Calculate Tangent Displacement Vector (Z_ni) ---
        // This is the same calculation as Algorithm 1's predictor step [cite: 1313, 1337]
        MatrixNxK PX_M = project_tangent(M, Xi);
        MatrixNxK drift_term = lambda_ * PX_M;

        double correction_coeff = C_nk_ * sigma_sq * dist_XM_sq / 2.0;
        MatrixNxK correction_term = correction_coeff * Xi;

        MatrixNxK dW = MatrixNxK::NullaryExpr(n_, k_, [&](){ return dist_(gen_); });
        MatrixNxK PX_dW = project_tangent(dW, Xi);
        MatrixNxK noise_term = sigma_ * dist_XM * PX_dW * sqrt_h;

        // Z_ni is the total displacement vector for the time step h
        MatrixNxK Z_ni = (drift_term - correction_term) * h + noise_term;

        // --- 2. Construct Skew-Symmetric Matrix W_ni ---
        // (Using the simplified construction W = A - A^T where A = Z * X^T)
        // Note: The paper uses a more complex formula[cite: 1325, 1339], but this
        // is a common and effective alternative construction.
        MatrixNxN A = Z_ni * Xi.transpose();
        MatrixNxN W_ni = A - A.transpose();   // Skew-symmetric (n x n) matrix

        // --- 3. Corrector Step (Cayley Transform) [cite: 1340] ---
        // Solve the linear system: (I - W/2) X_next = (I + W/2) X_i
        MatrixNxN left_matrix = Id_n - 0.5 * W_ni;
        MatrixNxK right_vector_part = (Id_n + 0.5 * W_ni) * Xi;

        // Use Eigen's robust linear solver (Householder QR decomposition)
        Eigen::HouseholderQR<MatrixNxN> qr(left_matrix);
        MatrixNxK X_next_i = qr.solve(right_vector_part);

        // Failsafe Check: If solver failed or result is non-finite
        if (!X_next_i.allFinite()) {
            new_particles[i] = Xi; // Keep previous state
        } else {
             // Optional: Check if still on manifold (numerical drift)
             MatrixKxK XtX = X_next_i.transpose() * X_next_i;
             MatrixKxK Id_k = MatrixKxK::Identity(k_, k_);
             if (!XtX.isApprox(Id_k, 1e-6)) {
                 // Re-project using SVD if Cayley transform drifted
                 new_particles[i] = project_manifold(X_next_i);
             } else {
                 new_particles[i] = X_next_i;
             }
        }
    }
    particles_ = std::move(new_particles); // Update particle states
}

// --- Solve Method ---
CBOResult KimCBOSolver::solve(double T, double h) {
    int num_steps = static_cast<int>(T / h);
    if (num_steps <= 0) return {};

    initialize_particles(); // Uses base class method

    std::vector<double> f_consensus_history(num_steps, std::numeric_limits<double>::quiet_NaN());

    // --- Initial State Calculation ---
    // calculate_consensus_stable() uses objective_ptr_
    MatrixNxK M_current = calculate_consensus_stable();
    // objective_ptr_->calculate_cost() is called implicitly
    double f_initial = objective_ptr_->calculate_cost(M_current);
    if (std::isfinite(f_initial)) {
        f_consensus_history[0] = f_initial;
    }

    // --- Main Loop ---
    for (int step = 0; step < num_steps - 1; ++step) {
        run_single_step(h); // Updates internal particles_

        M_current = calculate_consensus_stable();
        double f_val = objective_ptr_->calculate_cost(M_current);

        // Store result, fallback to previous if NaN
        f_consensus_history[step + 1] = std::isfinite(f_val) ? f_val : f_consensus_history[step];
    }

    // --- Final Cleanup (Handle NaNs/filling) ---
    // (Ensure this logic is robust as in previous versions)
    size_t first_valid_idx = 0;
    while (first_valid_idx < f_consensus_history.size() && !std::isfinite(f_consensus_history[first_valid_idx])) {
        first_valid_idx++;
    }
    if (first_valid_idx < f_consensus_history.size()) {
        if (first_valid_idx > 0) {
            std::fill(f_consensus_history.begin(), f_consensus_history.begin() + first_valid_idx, f_consensus_history[first_valid_idx]);
        }
        double last_valid = f_consensus_history[first_valid_idx];
         for(size_t i = first_valid_idx; i < f_consensus_history.size(); ++i) {
             if(std::isfinite(f_consensus_history[i])) {
                 last_valid = f_consensus_history[i];
             } else {
                 f_consensus_history[i] = last_valid;
             }
         }
    } else { // All NaNs case
         std::fill(f_consensus_history.begin(), f_consensus_history.end(), 0.0);
    }

    /*
    std::map<std::string, std::vector<double>> results;
    results["f_history"] = f_consensus_history;
    results["final_X"] = M_current;
    */

    CBOResult results;
    results.f_history = f_consensus_history;
    results.final_X = M_current;

    return results;
}