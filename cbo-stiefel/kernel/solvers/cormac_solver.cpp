// filename: cormac_solver.cpp

#include "solvers/cormac_solver.h" // Include the corresponding header
#include <cmath>
#include <numeric>
#include <algorithm> // For std::min, std::max
#include <limits>    // For numeric_limits
#include <vector>
#include <Eigen/LU>  // Required for solving linear systems in Cayley transform
#include <Eigen/QR>

// --- Constructor Definition (Matches Header) ---
CormacsCBOSolver::CormacsCBOSolver(int n, int k,
                                   BaseObjective* objective_ptr, // Pass C++ objective pointer
                                   int N_particles, double beta_val,
                                   // Lambda Adaptation
                                   double lambda_initial, double lambda_min, double lambda_max,
                                   double lambda_increase_factor, double lambda_decrease_factor,
                                   int lambda_adapt_interval, double lambda_stagnation_thresh,
                                   double lambda_convergence_thresh,
                                   // Sigma Adaptation
                                   double sigma_initial, double sigma_final, double sigma_max,
                                   double annealing_rate, int reheat_check_interval,
                                   int reheat_window, double reheat_threshold,
                                   double reheat_sigma_boost, bool reheat_lambda_reset,
                                   const std::string& backend)
    : BaseCBOSolver(n, k, objective_ptr, N_particles, beta_val, backend),
      // Initialize all adaptive members
      lambda_initial_(lambda_initial), sigma_initial_(sigma_initial), annealing_rate_(annealing_rate),
      lambda_min_(lambda_min), lambda_max_(lambda_max),
      lambda_increase_factor_(lambda_increase_factor), lambda_decrease_factor_(lambda_decrease_factor),
      lambda_adapt_interval_(lambda_adapt_interval), lambda_stagnation_thresh_(lambda_stagnation_thresh),
      lambda_convergence_thresh_(lambda_convergence_thresh),
      sigma_final_(sigma_final), sigma_max_(sigma_max),
      reheat_check_interval_(reheat_check_interval), reheat_window_(reheat_window),
      reheat_threshold_(reheat_threshold), reheat_sigma_boost_(reheat_sigma_boost),
      reheat_lambda_reset_(reheat_lambda_reset)
{
    // Base constructor handles objective_ptr_ storage and common initializations.
}

// --- Adaptive Helper: Annealing Schedule ---
double CormacsCBOSolver::get_sigma_annealed(double t, double T) const {
    if (T <= 0) return sigma_final_;
    // Use std::min/max for safety against large exponents
    double exponent = -std::min(700.0, annealing_rate_ * t / T);
    // Uses std::exp from <cmath>
    return std::max(sigma_final_, sigma_initial_ * std::exp(exponent));
}

// --- SDE Step Logic (Cayley Transform) ---
void CormacsCBOSolver::run_single_step(double h, double lambda_current, double sigma_current) {
    MatrixNxK M = calculate_consensus_stable(); // Consensus point for this step
    std::vector<MatrixNxK> new_particles(N_); // Buffer for updated particles
    double sigma_sq = sigma_current * sigma_current;
    double sqrt_h = std::sqrt(h);
    MatrixNxN Id_n = MatrixNxN::Identity(n_, n_); // n x n Identity matrix

    for (int i = 0; i < N_; ++i) {
        const MatrixNxK& Xi = particles_[i];

        // --- Predictor Step Components (Calculate Z_ni) ---

        // 1. Drift: lambda * P_X(X_bar - X_i)
        MatrixNxK drift_consensus_part = project_tangent(M - Xi, Xi);
        MatrixNxK drift_term = lambda_current * drift_consensus_part;

        // 2. Itô Correction Term: -C_nk * sigma^2 / 2 * X_i
        MatrixNxK correction_term = (C_nk_ * sigma_sq / 2.0) * Xi;

        // 3. Noise Term: sigma * P_X(dW) * sqrt(h)
        MatrixNxK dW = MatrixNxK::NullaryExpr(n_, k_, [&](){ return dist_(gen_); }); // Generate N(0,1) matrix
        MatrixNxK PX_dW = project_tangent(dW, Xi);
        MatrixNxK noise_term = sigma_current * PX_dW * sqrt_h;

        // Total displacement vector Z_ni
        MatrixNxK Z_ni = (drift_term - correction_term) * h + noise_term;

        // --- Corrector Step (Cayley Transform) ---

        // Construct Skew-Symmetric W_ni (n x n)
        MatrixNxN A = Z_ni * Xi.transpose();
        MatrixNxN W_ni = A - A.transpose();

        // Solve the linear system: (I - W/2) X_next = (I + W/2) X_i
        MatrixNxN left_matrix = Id_n - 0.5 * W_ni;
        MatrixNxK right_vector_part = (Id_n + 0.5 * W_ni) * Xi;

        // Use Eigen's robust linear solver (e.g., Householder QR decomposition)
        Eigen::HouseholderQR<MatrixNxN> qr(left_matrix);
        MatrixNxK X_next_i = qr.solve(right_vector_part);

        // --- CORRECTED Failsafe Check ---
        // Remove the qr.info() check. Rely on allFinite() for success indication.
        if (!X_next_i.allFinite()) {
        // --- End Correction ---

            // Option 1: Keep previous state (safer)
            new_particles[i] = Xi;
            // Option 2: Try SVD projection (might hide instability)
            // new_particles[i] = project_manifold(Xi + Z_ni); // Project the simple Euler step
        } else {
             // Optional: Check if still on manifold (numerical drift) and re-project if needed
             MatrixKxK XtX = X_next_i.transpose() * X_next_i;
             MatrixKxK Id_k = MatrixKxK::Identity(k_, k_);
             if (!XtX.isApprox(Id_k, 1e-6)) {
                 new_particles[i] = project_manifold(X_next_i); // SVD projection failsafe
             } else {
                 new_particles[i] = X_next_i;
             }
        }
    } // End particle loop
    particles_ = std::move(new_particles); // Update particle states
}

// --- Main Solve Loop (Includes Adaptation Logic) ---
std::map<std::string, std::vector<double>> CormacsCBOSolver::solve(double T, double h) {
    int num_steps = static_cast<int>(T / h);
    if (num_steps <= 0) return {};

    initialize_particles();

    // Initialize history arrays
    std::vector<double> f_consensus_history(num_steps, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> variance_history(num_steps, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> lambda_history(num_steps, lambda_initial_);
    std::vector<double> sigma_history(num_steps, sigma_initial_);

    double lambda_current = lambda_initial_;
    double sigma_current = sigma_initial_;
    double variance_prev_check = 1.0; // Initialize reasonably
    double f_consensus_min_window = std::numeric_limits<double>::infinity();

    // --- Initial State Calculation ---
    MatrixNxK M_current = calculate_consensus_stable();
    double f_val_init = objective_ptr_->calculate_cost(M_current); // Use C++ objective

    double current_variance_sum_sq_init = 0.0;
    for(const auto& p : particles_) { current_variance_sum_sq_init += (p - M_current).squaredNorm(); }
    double var_init = (N_ > 0) ? current_variance_sum_sq_init / N_ : 0.0;

    if (std::isfinite(f_val_init)) f_consensus_history[0] = f_val_init;
    if (std::isfinite(var_init)) variance_history[0] = var_init;

    variance_prev_check = std::isfinite(variance_history[0]) ? variance_history[0] : 1.0;
    if (variance_prev_check <= 1e-12) variance_prev_check = 1.0; // Avoid division by zero


    // --- Main Time-Stepping Loop ---
    for (int step = 0; step < num_steps; ++step) {
        double t = (step + 1) * h; // Time at the end of the step (for annealing calc)

        // 1. Parameter Adaptation (occurs at the START of the step's logic)
        sigma_current = get_sigma_annealed(t, T);

        // --- Reheating Logic ---
        if (step > 0 && step % reheat_check_interval_ == 0 && step >= reheat_window_) {
            int window_start = step - reheat_window_;
            double f_min_in_window = std::numeric_limits<double>::infinity();
            bool valid_window = false;
            for(int j = window_start; j < step; ++j) {
                if (std::isfinite(f_consensus_history[j])) {
                    f_min_in_window = std::min(f_min_in_window, f_consensus_history[j]);
                    valid_window = true;
                }
            }

            if (valid_window && std::isfinite(f_consensus_min_window)) {
                 double improvement = f_consensus_min_window - f_min_in_window;
                 if (improvement < reheat_threshold_) {
                     sigma_current = std::min(sigma_max_, sigma_current + reheat_sigma_boost_);
                     if (reheat_lambda_reset_) {
                         lambda_current = lambda_initial_;
                     }
                 }
                 f_consensus_min_window = std::min(f_consensus_min_window, f_min_in_window);
            } else if (valid_window) { // Initialize f_consensus_min_window if needed
                f_consensus_min_window = f_min_in_window;
            }
        } else if (step > 0 && std::isfinite(f_consensus_history[step-1])) {
             // Update min_window if not reheating
              f_consensus_min_window = std::min(f_consensus_min_window, f_consensus_history[step-1]);
        }


        // --- Lambda Adaptation Logic ---
        if (step > 0 && step % lambda_adapt_interval_ == 0) {
             int idx_prev = std::max(0, step - lambda_adapt_interval_);
             double var_prev = variance_history[idx_prev];
             double var_current_step_start = variance_history[step-1]; // Variance at end of last step

             if (std::isfinite(var_prev) && std::isfinite(var_current_step_start) && variance_prev_check > 1e-12) {
                 double relative_decrease = (var_prev - var_current_step_start) / variance_prev_check;
                 double time_interval = h * lambda_adapt_interval_;
                 double norm_stagnation_threshold = lambda_stagnation_thresh_ * time_interval;
                 double norm_convergence_threshold = lambda_convergence_thresh_ * time_interval;

                 if (relative_decrease < norm_stagnation_threshold) {
                     lambda_current = std::min(lambda_current * lambda_increase_factor_, lambda_max_);
                 } else if (relative_decrease > norm_convergence_threshold) {
                     lambda_current = std::max(lambda_current * lambda_decrease_factor_, lambda_min_);
                 }
                 variance_prev_check = var_current_step_start; // Update baseline
             }
        }

        lambda_history[step] = lambda_current;
        sigma_history[step] = sigma_current;

        // 2. Run CBO Step
        run_single_step(h, lambda_current, sigma_current);

        // 3. Store Metrics (based on state AFTER the update)
        M_current = calculate_consensus_stable();
        double f_val = objective_ptr_->calculate_cost(M_current);

        double current_variance_sum_sq = 0.0;
        for(const auto& p : particles_) { current_variance_sum_sq += (p - M_current).squaredNorm(); }
        double current_variance = (N_ > 0) ? current_variance_sum_sq / N_ : 0.0;

        // Store, using previous value as fallback if current is NaN
        f_consensus_history[step] = std::isfinite(f_val) ? f_val : f_consensus_history[std::max(0, step - 1)];
        variance_history[step] = std::isfinite(current_variance) ? current_variance : variance_history[std::max(0, step - 1)];

        // Update lambda baseline if needed (redundant here, done in adapt block)
        // variance_prev_check = variance_history[step];
    }

    // --- Final Cleanup and Packaging ---
    // (Helper function clean_history_np would be defined here or included from base)
    auto clean_history_vector = [](std::vector<double>& history) {
        size_t first_valid_idx = 0;
        while (first_valid_idx < history.size() && !std::isfinite(history[first_valid_idx])) {
            first_valid_idx++;
        }
        if (first_valid_idx < history.size()) {
            if (first_valid_idx > 0) {
                std::fill(history.begin(), history.begin() + first_valid_idx, history[first_valid_idx]);
            }
            double last_valid = history[first_valid_idx];
            for(size_t i = first_valid_idx; i < history.size(); ++i) {
                if(std::isfinite(history[i])) {
                    last_valid = history[i];
                } else {
                    history[i] = last_valid;
                }
            }
        } else { // All NaNs
             std::fill(history.begin(), history.end(), 0.0);
        }
    };

    clean_history_vector(f_consensus_history);
    clean_history_vector(variance_history);
    // Lambda/Sigma histories should not have NaNs if initialized correctly

    std::map<std::string, std::vector<double>> results;
    results["f_history"] = f_consensus_history;
    results["lambda_history"] = lambda_history;
    results["sigma_history"] = sigma_history;
    results["var_history"] = variance_history;

    return results;
}