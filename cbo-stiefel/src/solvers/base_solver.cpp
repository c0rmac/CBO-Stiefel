// filename: base_solver.cpp

#include "solvers/base_solver.h"
#include <Eigen/SVD>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <Eigen/Eigenvalues> // For potential future matrix sqrt
#include <iostream>
#include <cstdlib> // For srand
#include <ctime>   // For time

// Include torch headers only if the backend is compiled in
#ifdef USE_TORCH_BACKEND
#include <torch/torch.h>
#endif

// --- Constructor Definition ---

BaseCBOSolver::BaseCBOSolver(int n, int k,
                             BaseObjective* objective_ptr,
                             int N_particles, double beta_val, const std::string& backend, bool enforce_SOd)
    : n_(n), k_(k),
      objective_ptr_(objective_ptr),
      N_(N_particles), beta_(beta_val), backend_(backend), enforce_SOd_(enforce_SOd), particles_(N_particles)
{
    // srand(static_cast<unsigned int>(time(nullptr)));

    if (!objective_ptr_) {
        throw std::invalid_argument("Objective pointer cannot be null.");
    }
    if (k > n) {
        throw std::invalid_argument("k must be less than or equal to n.");
    }
    if (backend != "eigen" && backend != "torch") {
        throw std::invalid_argument("Backend must be 'eigen' or 'torch'.");
    }

    if (enforce_SOd_ && n_ != k_) {
        throw std::invalid_argument("SO(d) constraint (enforce_SOd=true) requires n == k.");
    }

    C_nk_ = (2.0 * n_ - k_ - 1.0) / 2.0;

    // Initialize RNG
    std::random_device rd;
    gen_ = std::mt19937(rd());
    dist_ = std::normal_distribution<>(0.0, 1.0);

#ifdef USE_TORCH_BACKEND
    // --- PyTorch Device Setup (Conditional) ---
    if (backend_ == "torch") {
        try {
            if (torch::has_mps()) {
                torch_device_ = torch::Device(torch::kMPS);
            } else {
                torch_device_ = torch::Device(torch::kCPU);
            }
            std::cout << "[BaseCBOSolver] Using PyTorch device: "
                      << (torch_device_.is_mps() ? "MPS" : "CPU") << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Warning: PyTorch backend failed to setup device (" << e.what()
                      << "). Falling back to Eigen backend." << std::endl;
            backend_ = "eigen";
        }
    }
#endif
    // std::cout << "[BaseCBOSolver] Final backend: " << backend_ << std::endl;
}

// --- Geometric Dispatchers ---

MatrixNxK BaseCBOSolver::project_tangent(const MatrixNxK& Z, const MatrixNxK& X) {
    // Keep using Eigen for tangent projection to avoid constant conversions
    return project_tangent_eigen(Z, X);
}

MatrixNxK BaseCBOSolver::project_manifold(const MatrixNxK& Z) {
    if (backend_ == "torch") {
#ifdef USE_TORCH_BACKEND
        return project_manifold_torch(Z); // Dispatch to Torch
#else
        return project_manifold_eigen(Z); // Fallback if compiled without Torch
#endif
    } else { // backend == "eigen"
        return project_manifold_eigen(Z); // Dispatch to Eigen
    }
}

// ======================================================================
// 1. STATIC EIGEN IMPLEMENTATIONS (CORE GEOMETRY)
// ======================================================================

/*
MatrixNxK BaseCBOSolver::project_tangent_eigen(const MatrixNxK& Z, const MatrixNxK& X) {
    if (!Z.allFinite() || !X.allFinite()) { return MatrixNxK::Zero(X.rows(), X.cols()); }

    MatrixKxK ZtX = Z.transpose() * X;
    MatrixNxN XXt = X * X.transpose();

    MatrixNxK projection = Z - 0.5 * (X * ZtX.transpose() + XXt * Z);

    if (!projection.allFinite()) { return MatrixNxK::Zero(X.rows(), X.cols()); }

    return projection;
}
*/
MatrixNxK BaseCBOSolver::project_tangent_eigen(const MatrixNxK& Z, const MatrixNxK& X) {
    if (!Z.allFinite() || !X.allFinite()) { return MatrixNxK::Zero(X.rows(), X.cols()); }

    // 1. Calculate the two k x k inner matrices
    MatrixKxK ZtX = Z.transpose() * X; // Z^T X
    MatrixKxK XtZ = X.transpose() * Z; // X^T Z

    // 2. Apply the correct formula: Z - 0.5 * (X(Z^T X) + X(X^T Z))
    // Note: X * ZtX is X(Z^T X)
    // Note: X * XtZ is X(X^T Z)
    MatrixNxK projection = Z - 0.5 * (X * ZtX + X * XtZ);

    if (!projection.allFinite()) { return MatrixNxK::Zero(X.rows(), X.cols()); }

    return projection;
}

MatrixNxK BaseCBOSolver::project_manifold_eigen(const MatrixNxK& Z) {
    // Note: This method is no longer static, it accesses self->enforce_SOd_

    // Failsafe generation (e.g., for initialize_particles)
    if (!Z.allFinite()) {
        MatrixNxK Z_rand = MatrixNxK::Random(Z.rows(), Z.cols());
        Eigen::JacobiSVD<MatrixNxK> svd_rand(Z_rand, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixNxK U = svd_rand.matrixU();
        MatrixNxK V = svd_rand.matrixV();
        MatrixNxK X_proj = U * V.transpose();

        // --- SO(d) Failsafe Check ---
        if (enforce_SOd_ && X_proj.determinant() < 0.0) {
            MatrixNxK U_flipped = U;
            U_flipped.col(k_-1) *= -1.0; // Flip the last column of U
            X_proj = U_flipped * V.transpose();
        }
        return X_proj;
    }

    // Main projection logic
    try {
        Eigen::JacobiSVD<MatrixNxK> svd(Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixNxK U = svd.matrixU();
        MatrixNxK V = svd.matrixV();
        MatrixNxK X_proj = U * V.transpose();

        // --- SO(d) Constraint Fix ---
        if (enforce_SOd_ && X_proj.determinant() < 0.0) {
            MatrixNxK U_flipped = U;
            U_flipped.col(k_-1) *= -1.0;
            X_proj = U_flipped * V.transpose();
        }
        // --- End Fix ---

         if (!X_proj.allFinite()) { throw std::runtime_error("Eigen SVD resulted in non-finite values."); }
        return X_proj;
    } catch (...) {
         // Failsafe (already handles SO(d) check from above)
         return project_manifold_eigen(MatrixNxK::Random(Z.rows(), Z.cols()));
    }
}

// ======================================================================
// 2. PYTORCH IMPLEMENTATIONS (CONDITIONAL ON USE_TORCH_BACKEND)
// ======================================================================

#ifdef USE_TORCH_BACKEND

// --- Torch Conversion Utilities ---

torch::Tensor BaseCBOSolver::eigen_to_torch(const MatrixNxK& eigen_mat) const {
    // Assuming MatrixXd is double
    return torch::from_blob(const_cast<double*>(eigen_mat.data()),
                            {eigen_mat.rows(), eigen_mat.cols()},
                            torch::kFloat64)
           .to(torch_device_);
}

MatrixNxK BaseCBOSolver::torch_to_eigen(const torch::Tensor& torch_tensor) const {
    torch::Tensor cpu_tensor = torch_tensor.to(torch::kCPU).contiguous();
    Eigen::Map<const MatrixNxK> eigen_map(cpu_tensor.data_ptr<double>(),
                                           cpu_tensor.size(0), cpu_tensor.size(1));
    return eigen_map;
}

// --- Torch Geometric Operations ---

torch::Tensor BaseCBOSolver::project_tangent_torch(const torch::Tensor& Z, const torch::Tensor& X) const {
     // NOTE: This function is declared in the header but its implementation is
     // simplified to use the Eigen implementation (via project_tangent_eigen)
     // in C++ for consistency, as moving Z and X back and forth is slower.
     // For a true PyTorch implementation, this would contain the tensor math.
     if (!torch::all(torch::isfinite(Z))) Z = torch::zeros_like(Z);
     if (!torch::all(torch::isfinite(X))) return torch::zeros_like(Z);

     torch::Tensor Z_T = Z.T;
     torch::Tensor X_T = X.T;
     torch::Tensor term1 = X.matmul(Z_T).matmul(X);
     torch::Tensor term2 = X.matmul(X_T).matmul(Z);
     torch::Tensor projection = Z - 0.5 * (term1 + term2);

     if (!torch::all(torch::isfinite(projection))) return torch::zeros_like(Z);
     return projection;
}


torch::Tensor BaseCBOSolver::project_manifold_torch_tensor(const torch::Tensor& Z) const {
    if (!torch::all(torch::isfinite(Z))) {
        torch::Tensor Z_rand = torch::randn_like(Z);
        auto [U_rand, S_rand, Vh_rand] = torch::linalg::svd(Z_rand, false);
        return U_rand.matmul(Vh_rand);
    }
    try {
        auto [U, S, Vh] = torch::linalg::svd(Z, false);
        torch::Tensor proj = U.matmul(Vh);
        if (!torch::all(torch::isfinite(proj))) {
             throw std::runtime_error("Torch SVD resulted in non-finite values.");
        }
        return proj;
    } catch (...) {
        throw std::runtime_error("PyTorch SVD failed during projection.");
    }
}


MatrixNxK BaseCBOSolver::project_manifold_torch(const MatrixNxK& Z_eigen) {
    if (!Z_eigen.allFinite()) {
        return project_manifold_eigen(MatrixNxK::Random(Z_eigen.rows(), Z_eigen.cols()));
    }
    try {
        torch::Tensor Z_torch = eigen_to_torch(Z_eigen);
        torch::Tensor proj_torch = project_manifold_torch_tensor(Z_torch);
        return torch_to_eigen(proj_torch);
    } catch (const std::runtime_error& e) {
        std::cerr << "Warning: Torch backend SVD failed (" << e.what() << "). Falling back to Eigen SVD." << std::endl;
        return project_manifold_eigen(Z_eigen);
    }
}

#endif // USE_TORCH_BACKEND


// ======================================================================
// 3. CORE CBO LOGIC (Backend-Agnostic State Management)
// ======================================================================

void BaseCBOSolver::initialize_particles() {
    particles_.resize(N_);
    for (int i = 0; i < N_; ++i) {
        // Use Eigen random generation
        MatrixNxK Z_eigen = MatrixNxK::NullaryExpr(n_, k_, [&](){ return dist_(gen_); });
        // Project using the selected backend via the dispatcher method
        particles_[i] = project_manifold(Z_eigen);
    }
}

MatrixNxK BaseCBOSolver::calculate_consensus_stable() {
    // --- Cost calculation uses the C++ objective directly ---
    VectorN costs(N_);
    bool any_finite_cost = false;
    for (int i = 0; i < N_; ++i) {
        costs[i] = objective_ptr_->calculate_cost(particles_[i]);
        if (std::isfinite(costs[i])) any_finite_cost = true;
    }
    // -----------------------------------------------------------

    MatrixNxK consensus_point = MatrixNxK::Zero(n_, k_); // Initialize consensus point

    if (!any_finite_cost) { // Fallback if ALL costs are invalid
        // Calculate simple mean of valid particles
        int valid_particle_count = 0;
        consensus_point.setZero(); // Ensure it starts at zero
        for(const auto& p : particles_) {
            if (p.allFinite()) {
                consensus_point += p;
                valid_particle_count++;
            }
        }
        // Corrected logic: perform division only if count > 0
        if (valid_particle_count > 0) {
            consensus_point /= valid_particle_count;
        }
        return consensus_point; // Return mean or zero
    }

    // Proceed with weighted average if at least one cost is finite
    double min_cost = costs.minCoeff();
    VectorN weights_unnormalized = (-beta_ * (costs.array() - min_cost)).exp();

    // Zero out weights for non-finite costs
    for(int i=0; i < N_; ++i) {
        if(!std::isfinite(costs[i])) weights_unnormalized[i] = 0.0;
    }

    double sum_weights = weights_unnormalized.sum();

    if (sum_weights < 1e-100 || !std::isfinite(sum_weights)) {
        // --- Fallback to empirical mean of valid particles ---
        int valid_fallback_count = 0;
        consensus_point.setZero();
        for(int i=0; i < N_; ++i) {
             if(std::isfinite(costs[i]) && particles_[i].allFinite()) {
                 consensus_point += particles_[i];
                 valid_fallback_count++;
             }
        }
        if (valid_fallback_count > 0) {
            consensus_point /= valid_fallback_count;
        }
        return consensus_point;
        // --- End Corrected Fallback ---

    } else { // Normal weighted average calculation
        VectorN normalized_weights = weights_unnormalized / sum_weights;
        consensus_point.setZero();
        for (int i = 0; i < N_; ++i) {
            if (normalized_weights[i] > 1e-12 && particles_[i].allFinite()) {
                consensus_point += normalized_weights[i] * particles_[i];
            }
        }

        // Final check for NaNs potentially introduced if all weights were near zero
        if (!consensus_point.allFinite()) {
            // Recompute mean robustly if final result is corrupted
            int valid_final_check_count = 0;
            consensus_point.setZero();
             for(int i=0; i < N_; ++i) {
                 if(std::isfinite(costs[i]) && particles_[i].allFinite()) {
                     consensus_point += particles_[i]; valid_final_check_count++;
                 }
            }
             // **FIXED LOGIC**: Use if/else instead of ternary
             if (valid_final_check_count > 0) {
                  consensus_point /= valid_final_check_count; // Divide in place
             }
             // Return consensus_point (which is either the mean or still zero)
             return consensus_point;
             // **END FIXED LOGIC**
        }
        return consensus_point; // Return normally computed consensus
    }
}

double BaseCBOSolver::randn() { return dist_(gen_); }