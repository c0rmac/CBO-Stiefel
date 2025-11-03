// filename: solvers/cbo_result.h

#ifndef CBO_RESULT_H
#define CBO_RESULT_H

#include <vector>
#include <Eigen/Core>

/**
 * @brief A general C++ struct to hold the results of a CBO run.
 *
 * This struct is "pure C++" and has no dependency on pybind11.
 * It is used as the return type for the core C++ solvers.
 * The Python binding layer (cbo_module.cpp) will be responsible
 * for converting this struct into a Python object (e.g., a dict).
 */
struct CBOResult {
    // History of the objective function value
    std::vector<double> f_history;

    // The final n x k matrix (the minimizer)
    Eigen::MatrixXd final_X;

    // Default constructor
    CBOResult() = default;
};

#endif // CBO_RESULT_H