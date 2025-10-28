// filename: cormac_solver.h

#ifndef CORMAC_SOLVER_H
#define CORMAC_SOLVER_H

#include "base_solver.h" // Include the base class structure
#include <map>
#include <string>
#include <vector>

// Define key initial parameters (optional, for clarity)
#define DEFAULT_CORMAC_N 200
#define DEFAULT_CORMAC_BETA 50.0

class CormacsCBOSolver : public BaseCBOSolver {
public:
    // --- Constructor Declaration (Includes all adaptive parameters) ---
    CormacsCBOSolver(int n, int k,
                     BaseObjective* objective_ptr, // Pass C++ objective pointer
                     int N_particles = DEFAULT_CORMAC_N,
                     double beta_val = DEFAULT_CORMAC_BETA,

                     // --- Lambda Adaptation Parameters ---
                     double lambda_initial = 1.0,
                     double lambda_min = 0.1,
                     double lambda_max = 100.0,
                     double lambda_increase_factor = 1.1,
                     double lambda_decrease_factor = 0.98,
                     int lambda_adapt_interval = 20,
                     double lambda_stagnation_thresh = 0.005,
                     double lambda_convergence_thresh = 0.3,

                     // --- Sigma Annealing/Reheating Parameters ---
                     double sigma_initial = 0.5,
                     double sigma_final = 0.001,
                     double sigma_max = 1.0,
                     double annealing_rate = 3.0,
                     int reheat_check_interval = 50,
                     int reheat_window = 100,
                     double reheat_threshold = 1e-4,
                     double reheat_sigma_boost = 0.2,
                     bool reheat_lambda_reset = true,

                     const std::string& backend = "eigen");

    // --- Overrides from BaseCBOSolver ---
    std::map<std::string, std::vector<double>> solve(double T, double h) override;
    std::string get_solver_name() const override { return "CormacsCBOSolver"; }

private:
    // --- Private Member Variables (Store adaptive parameters) ---
    double lambda_initial_;
    double lambda_min_;
    double lambda_max_;
    double lambda_increase_factor_;
    double lambda_decrease_factor_;
    int lambda_adapt_interval_;
    double lambda_stagnation_thresh_;
    double lambda_convergence_thresh_;

    double sigma_initial_;
    double sigma_final_;
    double sigma_max_;
    double annealing_rate_;
    int reheat_check_interval_;
    int reheat_window_;
    double reheat_threshold_;
    double reheat_sigma_boost_;
    bool reheat_lambda_reset_;

    // SDE update logic specific to Cormac's adaptive solver
    void run_single_step(double h, double lambda_current, double sigma_current);
    // Annealing schedule calculation
    double get_sigma_annealed(double t, double T) const;
};

#endif // CORMAC_SOLVER_H