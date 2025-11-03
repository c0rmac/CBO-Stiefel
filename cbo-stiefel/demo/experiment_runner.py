# filename: experiment_runner.py
import os

import numpy as np
import time
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import multiprocessing  # Import for setting start method

# --- Imports needed for the runner and worker ---
try:
    # 1. Get the current directory (demo/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Get the parent directory (the project root, which contains cbo_module.so)
    parent_dir = os.path.join(current_dir, '..')

    # 3. Insert the parent directory path to Python's module search path
    sys.path.insert(0, parent_dir)

    import cbo_module  # C++ kernel
except ImportError as e:
    print(f"Warning: Could not import necessary modules in ExperimentRunner file: {e}")


# ==============================================================================
# WORKER FUNCTION FOR PARALLEL EXECUTION
# ==============================================================================

def _run_single_trial_worker(args_tuple):
    """
    Worker function executed in parallel for a single CBO trial.
    Takes a tuple containing all necessary parameters.
    """
    # Unpack arguments
    solver_class_ref, objective_name, n, k, num_particles, T, dt, solver_params, objective_params, trial_id = args_tuple

    # --- Instantiate C++ Objective ---
    objective_cpp_instance = None
    try:
        # Dynamically instantiate the correct objective based on name
        if objective_name == "Ackley":
            objective_cpp_instance = cbo_module.AckleyObjective(n, k, **objective_params)
        elif objective_name == "WOPP":
            objective_cpp_instance = cbo_module.WOPPObjective(n, k, **objective_params)
        elif objective_name == "QAP":
            objective_cpp_instance = cbo_module.QAPObjective(n, k, **objective_params)
        elif objective_name == "GEP":
            objective_cpp_instance = cbo_module.GEPObjective(n, **objective_params)
        elif objective_name == "OPP":
            objective_cpp_instance = cbo_module.OPPObjective(n, **objective_params)
        elif objective_name == "RotationAveraging":
            objective_cpp_instance = cbo_module.RotationAveragingObjective(n, **objective_params)

        # 🌟 ADDED BLOCK FOR DRO-PCA
        elif objective_name == "DroPca":
            objective_cpp_instance = cbo_module.DroPcaObjective(n, k, **objective_params)

        else:
            raise ValueError(f"Unknown C++ objective name provided to worker: {objective_name}")

    except Exception as e:
        print(f"\n[Worker {trial_id} ERROR] Failed to create C++ objective '{objective_name}': {e}")
        num_steps_expected = int(T / dt) if dt > 0 else 1
        empty_history = np.full(num_steps_expected, np.nan)
        return {'f_history': empty_history, 'time': np.linspace(0, T, num_steps_expected), 'final_X': None,
                'trial_id': trial_id}

    # --- Instantiate C++ Solver ---
    try:
        solver = solver_class_ref(
            n=n,
            k=k,
            objective_ptr=objective_cpp_instance,  # Pass C++ object directly
            N_particles=num_particles,
            **solver_params
        )

        # --- Run Simulation ---
        # 🌟 Assumes solve() now returns a dict including 'final_X'
        results_dict = solver.solve(T, dt)

        # Ensure 'time' key exists
        if 'time' not in results_dict:
            num_steps = len(results_dict.get('f_history', []))
            results_dict['time'] = np.linspace(0, T, num_steps) if num_steps > 0 else np.array([0.0])

        # Add trial ID for tracking
        results_dict['trial_id'] = trial_id

        # Ensure 'final_X' key exists, even if it's None (e.g., if C++ code is old)
        if 'final_X' not in results_dict:
            results_dict['final_X'] = None

        return results_dict

    except Exception as e:
        print(f"\n[Worker {trial_id} ERROR] Solver '{solver_class_ref.__name__}' failed: {e}")
        num_steps_expected = int(T / dt) if dt > 0 else 1
        empty_history = np.full(num_steps_expected, np.nan)
        return {'f_history': empty_history, 'time': np.linspace(0, T, num_steps_expected), 'final_X': None,
                'trial_id': trial_id}


# ==============================================================================
# EXPERIMENT RUNNER CLASS
# ==============================================================================

class ExperimentRunner:
    """
    Manages the execution of multiple CBO trials in parallel, 
    tracks the best minimizer X, and C++ objectives.
    """

    def __init__(self,
                 solver_class,
                 objective_name: str,
                 n: int, k: int, num_particles: int,
                 T: float, dt: float, trials: int,
                 success_tolerance: float,
                 solver_params: dict,
                 objective_params: dict = None):

        self.solver_class = solver_class
        self.objective_name = objective_name
        self.n = n
        self.k = k
        self.num_particles = num_particles
        self.T = T
        self.dt = dt
        self.trials = trials
        self.success_tolerance = success_tolerance
        self.solver_params = solver_params
        self.objective_params = objective_params if objective_params is not None else {}

        # Get objective metadata
        try:
            # (Instantiating all objectives as in your original file)
            if self.objective_name == "Ackley":
                temp_obj = cbo_module.AckleyObjective(n, k, **self.objective_params)
            elif self.objective_name == "WOPP":
                temp_obj = cbo_module.WOPPObjective(n, k, **self.objective_params)
            elif self.objective_name == "QAP":
                temp_obj = cbo_module.QAPObjective(n, k, **self.objective_params)
            elif self.objective_name == "GEP":
                temp_obj = cbo_module.GEPObjective(n, **self.objective_params)
            elif self.objective_name == "OPP":
                temp_obj = cbo_module.OPPObjective(n, **self.objective_params)
            elif self.objective_name == "RotationAveraging":
                temp_obj = cbo_module.RotationAveragingObjective(n, **self.objective_params)
            elif self.objective_name == "DroPca":
                temp_obj = cbo_module.DroPcaObjective(n, k, **self.objective_params)
            else:
                temp_obj = None

            self.true_minimum = temp_obj.get_true_minimum() if temp_obj else None
        except Exception:
            self.true_minimum = None

            # Results storage
        self.results_list = []
        self.time_points = None
        self.total_runtime = 0.0
        self.all_f_M_dynamics = None
        self.asymptotic_values = None
        self.success_count = 0

        # 🌟 --- NEW MEMBERS --- 🌟
        # To store the best matrix found
        self.best_final_value = np.inf
        self.best_final_X = None

    def run_trials_parallel(self):
        """
        Orchestrates the parallel execution of all configured trials using multiprocessing.Pool.
        """
        solver_name = self.solver_class.__name__
        print("=" * 60)
        print(f"--- Running Parallel Experiment: {solver_name} ({self.objective_name}) on V({self.k},{self.n}) ---")
        print(f"Required Trials: {self.trials}")
        try:
            num_workers = cpu_count()
            print(f"Using {num_workers} CPU cores.")
        except NotImplementedError:
            num_workers = 1
            print("Warning: Could not determine CPU count. Using 1 worker.")
        print("=" * 60)

        args_list = [
            (self.solver_class, self.objective_name, self.n, self.k,
             self.num_particles, self.T, self.dt, self.solver_params,
             self.objective_params, trial_id)
            for trial_id in range(self.trials)
        ]

        start_time_total = time.time()

        try:
            with Pool(num_workers) as pool:
                results_iterator = tqdm(
                    pool.imap_unordered(_run_single_trial_worker, args_list),
                    total=self.trials,
                    desc=f"Running {solver_name} Trials",
                    unit="trial"
                )
                self.results_list = list(results_iterator)
        except Exception as e:
            print(f"\nFATAL ERROR during multiprocessing pool execution: {e}")

        end_time_total = time.time()
        self.total_runtime = end_time_total - start_time_total

        self._aggregate_results()
        self._print_summary(solver_name)

    def _aggregate_results(self):
        """Processes the list of dictionaries, identifies the best minimizer X."""
        if not self.results_list:
            print("Warning: No results collected from workers.")
            return

        # Filter out potentially failed trials
        valid_results = [res for res in self.results_list if
                         res and res.get('f_history') is not None and len(res['f_history']) > 0]

        if not valid_results:
            print("Warning: No valid results found after filtering worker outputs.")
            return

        self.all_f_M_dynamics = []
        final_values_list = []

        # 🌟 --- NEW LOGIC --- 🌟
        # Find the best matrix X as we iterate

        self.best_final_value = np.inf
        self.best_final_X = None

        # Use first valid one as reference for length
        ref_len = len(valid_results[0]['f_history'])

        for res in valid_results:
            # Get history
            f_hist = res['f_history']
            if len(f_hist) != ref_len:
                f_hist = np.full(ref_len, np.nan)  # Pad if lengths mismatch
            self.all_f_M_dynamics.append(f_hist)

            # Get final cost
            final_cost = f_hist[-1]
            final_values_list.append(final_cost)

            # Check if this is the new best solution
            if np.isfinite(final_cost) and final_cost < self.best_final_value:
                # Check if the matrix X was returned
                if res.get('final_X') is not None:
                    self.best_final_value = final_cost
                    self.best_final_X = res['final_X']  # Store the matrix

        self.asymptotic_values = np.array(final_values_list)

        # Set time points
        self.time_points = valid_results[0].get('time', np.linspace(0, self.T, ref_len))
        if len(self.time_points) != ref_len:
            self.time_points = np.linspace(0, self.T, ref_len)

        if self.asymptotic_values.size > 0:
            if self.true_minimum is not None and not np.isnan(self.true_minimum):
                self.success_count = np.sum(self.asymptotic_values < self.success_tolerance)
            else:
                self.success_count = 0
        else:
            self.success_count = 0

    def _print_summary(self, solver_name):
        """Prints the final summary statistics."""
        print("\n" + "=" * 60)
        print(f"EXPERIMENT SUMMARY ({solver_name} ({self.objective_name}) on V({self.k},{self.n}))")
        print(f"Total Runtime: {self.total_runtime:.2f} seconds.")

        if self.asymptotic_values is not None and self.asymptotic_values.size > 0:
            if self.true_minimum is not None:
                print(f"True Minimum: {self.true_minimum}")
            print(
                f"Success Rate (<{self.success_tolerance:.1e}): {self.success_count}/{self.trials} ({self.success_count / self.trials * 100:.2f}%)")
            print(f"Mean Final f(M): {np.mean(self.asymptotic_values):.6e}")
            print(f"Std Dev Final f(M): {np.std(self.asymptotic_values):.6e}")
            print(f"Min Final f(M):  {np.min(self.asymptotic_values):.6e}")
            print(f"Max Final f(M):  {np.max(self.asymptotic_values):.6e}")

            # 🌟 --- NEW --- 🌟
            if self.best_final_X is not None:
                print(f"---")
                print(f"🏆 Best minimizer X (shape {self.best_final_X.shape}) was found.")
                print(f"   It achieved the global min value of: {self.best_final_value:.6e}")
            else:
                print(f"---")
                print(f"⚠️ Warning: Best minimizer X was not returned from C++.")
                print(f"   The C++ solve() method must return a dict key 'final_X'.")

        else:
            print("Warning: No valid asymptotic data collected across all trials.")
        print("=" * 60)

    def get_results_for_plotting(self):
        """Returns the aggregated data needed for generating plots."""
        # (This function remains the same, just creates a temp objective)
        plotting_objective_instance = None
        try:
            if self.objective_name == "Ackley":
                plotting_objective_instance = cbo_module.AckleyObjective(self.n, self.k, **self.objective_params)
            elif self.objective_name == "WOPP":
                plotting_objective_instance = cbo_module.WOPPObjective(self.n, self.k, **self.objective_params)
            elif self.objective_name == "QAP":
                plotting_objective_instance = cbo_module.QAPObjective(self.n, self.k, **self.objective_params)
            elif self.objective_name == "GEP":
                plotting_objective_instance = cbo_module.GEPObjective(self.n, **self.objective_params)
            elif self.objective_name == "OPP":
                plotting_objective_instance = cbo_module.OPPObjective(self.n, **self.objective_params)
            elif self.objective_name == "RotationAveraging":
                plotting_objective_instance = cbo_module.RotationAveragingObjective(self.n, **self.objective_params)
            elif self.objective_name == "DroPca":
                plotting_objective_instance = cbo_module.DroPcaObjective(self.n, self.k, **self.objective_params)
        except Exception:
            class MockObjective:
                def __init__(self, n, k): self.n, self.k = n, k

                def get_true_minimum(self): return 0.0

            plotting_objective_instance = MockObjective(self.n, self.k)

        return self.time_points, self.all_f_M_dynamics, self.asymptotic_values, plotting_objective_instance

    # 🌟 --- NEW GETTER METHOD --- 🌟
    def get_best_minimizer(self):
        """Returns the best final matrix X (the minimizer) found across all trials."""
        return self.best_final_X