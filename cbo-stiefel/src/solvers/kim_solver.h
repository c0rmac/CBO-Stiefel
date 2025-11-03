// filename: kim_solver.h

#ifndef KIM_SOLVER_H
#define KIM_SOLVER_H

#include "solvers/base_solver.h" // Include the abstract base class structure
#include <map>
#include <string>
#include <vector>

// Forward declare BaseObjective if needed, though base_solver.h includes it now
// class BaseObjective;

class KimCBOSolver : public BaseCBOSolver {
public:
    // --- Constructor Declaration (Updated) ---
    // Now takes BaseObjective* instead of std::function
    KimCBOSolver(int n, int k,
                 BaseObjective* objective_ptr, // Pass C++ objective pointer
                 int N_particles = 50,
                 double beta_val = 5000.0,
                 double lambda_val = 1.0,
                 double sigma_val = 0.5,
                 const std::string& backend = "eigen",
                 bool enforce_SOd = false);

    // --- Overrides from BaseCBOSolver ---
    CBOResult solve(double T, double h) override;
    std::string get_solver_name() const override { return "KimCBOSolver"; }

private:
    // SDE update logic (defined in .cpp)
    void run_single_step(double h);

    // Fixed Parameters
    double lambda_;
    double sigma_;
};

#endif // KIM_SOLVER_H