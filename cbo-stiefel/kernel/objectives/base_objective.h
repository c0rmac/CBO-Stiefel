// filename: kernel/objectives/base_objective.h

#ifndef BASE_OBJECTIVE_H
#define BASE_OBJECTIVE_H

#include <Eigen/Core>
#include <stdexcept>
#include <functional> // Used implicitly by the solvers
#include <cstdlib> // For srand
#include <ctime>   // For time

using MatrixNxK = Eigen::MatrixXd;

class BaseObjective {
public:
    // --- Constructor ---
    BaseObjective(int n, int k) : n_(n), k_(k) {
        if (k > n) {
            throw std::invalid_argument("k must be less than or equal to n.");
        }
        nk_ = static_cast<double>(n * k);
        // srand(static_cast<unsigned int>(time(nullptr)));
    }

    virtual ~BaseObjective() = default;

    // --- Core Interface (Pure Virtual Function) ---
    // Every objective MUST implement the cost calculation.
    virtual double calculate_cost(const MatrixNxK& X) const = 0;

    // --- Function Call Operator (Allows solvers to call objective(X)) ---
    double operator()(const MatrixNxK& X) const {
        return calculate_cost(X);
    }

    // --- Interface Getters ---
    virtual const MatrixNxK& get_minimizer() const = 0;

    // Virtual member to hold the known minimum value (e.g., 0.0 for Ackley)
    virtual double get_true_minimum() const = 0;

protected:
    int n_;
    int k_;
    double nk_; // Pre-calculated n * k
};

#endif // BASE_OBJECTIVE_H