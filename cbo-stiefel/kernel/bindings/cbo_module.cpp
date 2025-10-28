// filename: kernel/bindings/cbo_module.cpp

#include "../kernel/solvers/base_solver.h" // Includes BaseObjective forward declaration
#include "../kernel/solvers/kim_solver.h"
#include "../kernel/solvers/cormac_solver.h"
#include "../kernel/objectives/base_objective.h" // Include base objective header
#include "../kernel/objectives/ackley_objective.h" // Include concrete objective
#include "../kernel/objectives/wopp_objective.h" // Include WOPP Objective
#include "../kernel/objectives/qap_objective.h"  // Include QAP Objective

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h> // Keep for potential future use

namespace py = pybind11;

// --- Pybind11 Module Definition ---

PYBIND11_MODULE(cbo_module, m) {
    m.doc() = "Pybind11 module for accelerated CBO solvers and objectives (Tessera Project).";

    // --- 1. Bind BaseObjective FIRST ---
    // Needed before binding solvers that take it as an argument
    py::class_<BaseObjective>(m, "BaseObjective")
        .def("get_true_minimum", &BaseObjective::get_true_minimum)
        .def("__call__", &BaseObjective::operator())
        .def("__repr__", [](const BaseObjective& ) { return "<BaseObjective (C++)>"; });

    // --- 2. Bind AckleyObjective (Inherits from BaseObjective) ---
    py::class_<AckleyObjective, BaseObjective>(m, "AckleyObjective")
        .def(py::init<int, int, double, double, double>(),
             py::arg("n"), py::arg("k"),
             py::arg("a") = AckleyObjective::DEFAULT_A,
             py::arg("b") = AckleyObjective::DEFAULT_B,
             py::arg("c") = AckleyObjective::DEFAULT_C)
        .def("get_minimizer", &AckleyObjective::get_minimizer, py::return_value_policy::reference_internal);

    // --- 3.1 Bind BaseCBOSolver SECOND ---
    // Must be bound before KimCBOSolver and CormacsCBOSolver
    py::class_<BaseCBOSolver>(m, "BaseCBOSolver")
        // No constructor binding as it's abstract/protected constructor logic
        .def("get_solver_name", &BaseCBOSolver::get_solver_name)
        .def("__repr__", [](const BaseCBOSolver& a) {
            // Check actual type to provide more info
            if (dynamic_cast<const KimCBOSolver*>(&a)) return "<KimCBOSolver (C++)>";
            if (dynamic_cast<const CormacsCBOSolver*>(&a)) return "<CormacsCBOSolver (C++)>";
            return "<BaseCBOSolver (C++, abstract)>";
        });

    // --- 3.2 Bind WOPPObjective (NEW BINDING) ---
    // Implements the cost function f(X) = 0.5 * ||A @ X @ C - B||_F^2
    py::class_<WOPPObjective, BaseObjective>(m, "WOPPObjective")
        .def(py::init<int, int>(), // Constructor takes only n and k
             py::arg("n"), py::arg("k"),
             "Initializes the WOPP objective, generating random coefficient matrices (A, B, C) "
             "such that the true minimum is 0.")
        .def("get_true_minimum", &WOPPObjective::get_true_minimum);

    // --- 3.3 Bind QAPObjective (NEW BINDING) ---
    // Implements the cost function f(X) = tr(A X B X.T)
    py::class_<QAPObjective, BaseObjective>(m, "QAPObjective")
        .def(py::init<int, int>(), // Constructor takes only n and k (matrices A and B are randomly generated internally)
             py::arg("n"), py::arg("k"),
             "Initializes the QAP objective, generating random symmetric matrices A and B.")
        .def("get_true_minimum", &QAPObjective::get_true_minimum); // Expose the analytically calculated minimum

    // --- 4. Bind KimCBOSolver (Inherits from BaseCBOSolver) ---
    py::class_<KimCBOSolver, BaseCBOSolver>(m, "KimCBOSolver") // Specify Inheritance
        .def(py::init<int, int,
                      BaseObjective*, // Expects a pointer to BaseObjective
                      int, double, double, double, const std::string&>(),
             py::arg("n"), py::arg("k"),
             py::arg("objective_ptr"),
             py::arg("N_particles") = 50, py::arg("beta_val") = 5000.0,
             py::arg("lambda_val") = 1.0, py::arg("sigma_val") = 0.5,
             py::arg("backend") = "eigen",
             py::keep_alive<1, 3>()) // Keep objective_ptr alive
        .def("solve", &KimCBOSolver::solve, py::call_guard<py::gil_scoped_release>());
        // get_solver_name is inherited

    // --- 5. Bind CormacsCBOSolver (Inherits from BaseCBOSolver) ---
    py::class_<CormacsCBOSolver, BaseCBOSolver>(m, "CormacsCBOSolver") // Specify Inheritance
        .def(py::init<int, int,
                      BaseObjective*, // Expects a pointer to BaseObjective
                      int, double,
                      // Lambda Adaptive
                      double, double, double, double, double, int, double, double,
                      // Sigma Adaptive
                      double, double, double, double, int, int, double, double, bool,
                      const std::string&>(),
             py::arg("n"), py::arg("k"),
             py::arg("objective_ptr"),
             py::arg("N_particles") = DEFAULT_CORMAC_N, py::arg("beta_val") = DEFAULT_CORMAC_BETA,
             // Lambda args
             py::arg("lambda_initial") = 1.0, py::arg("lambda_min") = 0.1, py::arg("lambda_max") = 100.0,
             py::arg("lambda_increase_factor") = 1.1, py::arg("lambda_decrease_factor") = 0.98,
             py::arg("lambda_adapt_interval") = 20, py::arg("lambda_stagnation_thresh") = 0.005,
             py::arg("lambda_convergence_thresh") = 0.3,
             // Sigma args
             py::arg("sigma_initial") = 0.5, py::arg("sigma_final") = 0.001, py::arg("sigma_max") = 1.0,
             py::arg("annealing_rate") = 3.0, py::arg("reheat_check_interval") = 50,
             py::arg("reheat_window") = 100, py::arg("reheat_threshold") = 1e-4,
             py::arg("reheat_sigma_boost") = 0.2, py::arg("reheat_lambda_reset") = true,
             py::arg("backend") = "eigen",
             py::keep_alive<1, 3>()) // Keep objective_ptr alive
        .def("solve", &CormacsCBOSolver::solve, py::call_guard<py::gil_scoped_release>());
        // get_solver_name is inherited

} // End PYBIND11_MODULE