// filename: kim_solver.cpp

#include "solvers/kim_solver.h" // Include the corresponding header
#include <cmath>
#include <numeric>
#include <limits> // For numeric_limits

// --- Constructor Definition (Matches Header) ---
KimCBOSolver::KimCBOSolver(int n, int k,
                           BaseObjective* objective_ptr, // Changed parameter
                           int N_particles, double beta_val, double lambda_val, double sigma_val,
                           const std::string& backend)
    // Pass objective_ptr to the BaseCBOSolver constructor
    : BaseCBOSolver(n, k, objective_ptr, N_particles, beta_val, backend),
      lambda_(lambda_val), sigma_(sigma_val) {}

// --- Core SDE Step Logic ---
void KimCBOSolver::run_single_step(double h) {
    // calculate_consensus_stable() now uses objective_ptr_ internally
    MatrixNxK M = calculate_consensus_stable();
    std::vector<MatrixNxK> new_particles(N_);
    double sigma_sq = sigma_ * sigma_;
    double sqrt_h = std::sqrt(h);

    for (int i = 0; i < N_; ++i) {
        const MatrixNxK& Xi = particles_[i];
        double dist_XM_sq = (Xi - M).squaredNorm(); // Squared Frobenius norm

        // --- Predictor Step ---
        MatrixNxK PX_M = project_tangent(M, Xi); // Uses base class dispatcher
        MatrixNxK drift_term = lambda_ * PX_M;

        double correction_coeff = C_nk_ * sigma_sq * dist_XM_sq / 2.0;
        MatrixNxK correction_term = correction_coeff * Xi;

        MatrixNxK dW = MatrixNxK::NullaryExpr(n_, k_, [&](){ return dist_(gen_); });
        MatrixNxK PX_dW = project_tangent(dW, Xi); // Uses base class dispatcher

        double dist_XM = std::sqrt(dist_XM_sq);
        MatrixNxK noise_term = sigma_ * dist_XM * PX_dW * sqrt_h;

        // Euler-Maruyama Update
        MatrixNxK X_pred = Xi + (drift_term - correction_term) * h + noise_term;

        // --- Corrector Step ---
        new_particles[i] = project_manifold(X_pred); // Uses base class dispatcher
    }
    particles_ = std::move(new_particles); // Update particle states
}

// --- Solve Method ---
std::map<std::string, std::vector<double>> KimCBOSolver::solve(double T, double h) {
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

    std::map<std::string, std::vector<double>> results;
    results["f_history"] = f_consensus_history;
    // Optionally add final M (converted to vector<double>?) or other metrics

    return results;
}