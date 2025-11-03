// filename: solvers/base_solver.h

#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include <vector>
#include <functional>
#include <map>
#include <string>
#include <stdexcept>
#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>
#include <memory>
#include "cbo_result.h"

class BaseObjective;

#ifdef USE_TORCH_BACKEND
#include <torch/torch.h>
#endif

using MatrixNxK = Eigen::MatrixXd;
using MatrixKxK = Eigen::MatrixXd;
using MatrixNxN = Eigen::MatrixXd;
using VectorN = Eigen::VectorXd;

class BaseCBOSolver {
public:
    BaseCBOSolver(int n, int k,
                  BaseObjective* objective_ptr,
                  int N_particles, double beta_val,
                  const std::string& backend = "eigen",
                  bool enforce_SOd = false);

    virtual ~BaseCBOSolver() = default;

    virtual CBOResult solve(double T, double h) = 0;
    virtual std::string get_solver_name() const = 0;

protected:
    MatrixNxK project_tangent(const MatrixNxK& Z, const MatrixNxK& X);
    MatrixNxK project_manifold(const MatrixNxK& Z);

    void initialize_particles();
    MatrixNxK calculate_consensus_stable();
    double randn();

    // --- Static Helper Implementations (Defined in base_solver.cpp) ---
    static MatrixNxK project_tangent_eigen(const MatrixNxK& Z, const MatrixNxK& X);

    // --- NON-STATIC (THE FIX) ---
    // Removed 'static' because it needs access to n_, k_, and enforce_SOd_
    MatrixNxK project_manifold_eigen(const MatrixNxK& Z);

#ifdef USE_TORCH_BACKEND
    MatrixNxK project_manifold_torch(const MatrixNxK& Z_eigen);
    MatrixNxK torch_to_eigen(const torch::Tensor& torch_tensor) const;
    torch::Tensor eigen_to_torch(const MatrixNxK& eigen_mat) const;
    torch::Tensor project_tangent_torch(const torch::Tensor& Z, const torch::Tensor& X) const;
    torch::Tensor project_manifold_torch_tensor(const torch::Tensor& Z) const;
#endif

    // --- Member Variables ---
    int n_;
    int k_;
    BaseObjective* objective_ptr_;
    int N_;
    double beta_;
    double C_nk_;
    std::string backend_;
    bool enforce_SOd_; // Flag for SO(d)

    std::vector<MatrixNxK> particles_;
    std::mt19937 gen_;
    std::normal_distribution<> dist_;

#ifdef USE_TORCH_BACKEND
    torch::Device torch_device_ = torch::kCPU;
#endif
};

#include "../objectives/base_objective.h"

#endif // BASE_SOLVER_H