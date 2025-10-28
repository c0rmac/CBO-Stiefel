// filename: base_solver.h

#ifndef BASE_SOLVER_H
#define BASE_SOLVER_H

#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>
#include <memory> // For std::unique_ptr

// Forward declare the objective base class to avoid circular dependency
// if objective headers include base_solver.h
class BaseObjective;

// --- PyTorch C++ API Includes (Conditional) ---
#ifdef USE_TORCH_BACKEND
#include <torch/torch.h>
#endif

// Define common Eigen types
using MatrixNxK = Eigen::MatrixXd;
using MatrixKxK = Eigen::MatrixXd;
using MatrixNxN = Eigen::MatrixXd;
using VectorN = Eigen::VectorXd;

class BaseCBOSolver {
public:
    // --- Constructor Updated ---
    // Now takes a pointer to a BaseObjective instance.
    // Ownership can be managed via std::unique_ptr or shared_ptr if needed,
    // but a raw pointer is simple if lifetime is managed externally.
    BaseCBOSolver(int n, int k,
                  BaseObjective* objective_ptr, // Pass a pointer to the C++ objective
                  int N_particles, double beta_val,
                  const std::string& backend = "eigen");

    virtual ~BaseCBOSolver() = default;

    // --- Abstract Methods ---
    virtual std::map<std::string, std::vector<double>> solve(double T, double h) = 0;
    virtual std::string get_solver_name() const = 0;

protected:
    // --- Geometric Dispatchers ---
    MatrixNxK project_tangent(const MatrixNxK& Z, const MatrixNxK& X);
    MatrixNxK project_manifold(const MatrixNxK& Z);

    // --- Core CBO Logic ---
    void initialize_particles();
    MatrixNxK calculate_consensus_stable(); // Implementation will change
    double randn();

    // --- Static Helper Implementations (Defined in base_solver.cpp) ---
    static MatrixNxK project_tangent_eigen(const MatrixNxK& Z, const MatrixNxK& X);
    static MatrixNxK project_manifold_eigen(const MatrixNxK& Z);

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
    BaseObjective* objective_ptr_; // Store pointer to the C++ objective
    int N_;
    double beta_;
    double C_nk_;
    std::string backend_;

    std::vector<MatrixNxK> particles_;
    std::mt19937 gen_;
    std::normal_distribution<> dist_;

#ifdef USE_TORCH_BACKEND
    torch::Device torch_device_ = torch::kCPU;
#endif
};

// --- Include Objective Base Header AFTER forward declaration ---
// This is necessary if BaseObjective needs MatrixNxK type defined here.
#include "../objectives/base_objective.h"

#endif // BASE_SOLVER_H