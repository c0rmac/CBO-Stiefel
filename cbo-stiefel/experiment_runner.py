import numpy as np
import time
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import multiprocessing # Import for setting start method

# --- Imports needed for the runner and worker ---
# Assumes these modules are accessible (e.g., via sys.path or installation)
try:
    import cbo_module # C++ kernel
    # Need objective classes to instantiate them in the worker
    # Import BaseSolver for type hinting if desired (optional)
    # from base_solver import BaseCBOSolver
except ImportError as e:
    print(f"Warning: Could not import necessary modules in ExperimentRunner file: {e}")
    # Allow the script to define the class, but execution might fail later.

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
        # Add elif blocks here for other C++ objectives (WOPP, QAP)
        elif objective_name == "WOPP":
            objective_cpp_instance = cbo_module.WOPPObjective(n, k, **objective_params)
        elif objective_name == "QAP":
            objective_cpp_instance = cbo_module.QAPObjective(n, k, **objective_params)
        else:
             raise ValueError(f"Unknown C++ objective name provided to worker: {objective_name}")

    except Exception as e:
        print(f"\n[Worker {trial_id} ERROR] Failed to create C++ objective '{objective_name}': {e}")
        # Return minimal failure data
        num_steps_expected = int(T/dt) if dt > 0 else 1
        empty_history = np.full(num_steps_expected, np.nan)
        return {'f_history': empty_history, 'time': np.linspace(0, T, num_steps_expected), 'final_M': None, 'trial_id': trial_id}

    # --- Instantiate C++ Solver ---
    try:
        solver = solver_class_ref(
            n=n,
            k=k,
            objective_ptr=objective_cpp_instance, # Pass C++ object directly
            N_particles=num_particles,
            **solver_params # Unpack beta, lambda, sigma, backend etc.
        )

        # --- Run Simulation ---
        results_dict_cpp = solver.solve(T, dt) # Returns map<string, vector<double>>

        # Convert C++ map result to Python dict
        results = dict(results_dict_cpp)
        # Ensure 'time' key exists
        if 'time' not in results:
             num_steps = len(results.get('f_history', []))
             results['time'] = np.linspace(0, T, num_steps) if num_steps > 0 else np.array([0.0])
        # Add trial ID for tracking
        results['trial_id'] = trial_id

    except Exception as e:
        print(f"\n[Worker {trial_id} ERROR] Solver '{solver_class_ref.__name__}' failed: {e}")
        num_steps_expected = int(T/dt) if dt > 0 else 1
        empty_history = np.full(num_steps_expected, np.nan)
        results = {'f_history': empty_history, 'time': np.linspace(0, T, num_steps_expected), 'final_M': None, 'trial_id': trial_id}

    return results


# ==============================================================================
# EXPERIMENT RUNNER CLASS
# ==============================================================================

class ExperimentRunner:
    """
    Manages the execution of multiple CBO trials in parallel using C++ solvers
    and C++ objectives defined in the cbo_module.
    """
    def __init__(self,
                 solver_class,          # e.g., cbo_module.KimCBOSolver
                 objective_name: str,   # e.g., "Ackley"
                 n: int, k: int, num_particles: int,
                 T: float, dt: float, trials: int,
                 success_tolerance: float,
                 solver_params: dict,  # Dict with beta, lambda, sigma, backend etc.
                 objective_params: dict = None): # Optional dict for objective params
        """
        Initializes the experiment configuration.
        """
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

        # Get objective metadata (like true minimum) if needed for summary
        # We need to instantiate it temporarily here for this info
        try:
             if self.objective_name == "Ackley":
                 temp_obj = cbo_module.AckleyObjective(n, k, **self.objective_params)
                 self.true_minimum = temp_obj.get_true_minimum()
             # Add elif for other objectives...
             elif self.objective_name == "WOPP":
                 temp_obj = cbo_module.WOPPObjective(n, k, **self.objective_params)
                 self.true_minimum = temp_obj.get_true_minimum()
             elif self.objective_name == "QAP":
                 temp_obj = cbo_module.QAPObjective(n, k, **self.objective_params)
                 self.true_minimum = temp_obj.get_true_minimum()
             else:
                 self.true_minimum = None # Unknown
        except Exception:
             self.true_minimum = None # Failed to get info

        # Results storage
        self.results_list = []
        self.time_points = None
        self.total_runtime = 0.0
        self.all_f_M_dynamics = None
        self.asymptotic_values = None
        self.success_count = 0


    def run_trials_parallel(self):
        """
        Orchestrates the parallel execution of all configured trials using multiprocessing.Pool.
        """
        solver_name = self.solver_class.__name__
        print("=" * 60)
        print(f"--- Running Parallel Experiment: {solver_name} ({self.objective_name}) on V({self.k},{self.n}) ---")
        print(f"Required Trials: {self.trials}")
        # Try getting cpu_count safely
        try:
             num_workers = cpu_count()
             print(f"Using {num_workers} CPU cores.")
        except NotImplementedError:
             num_workers = 1
             print("Warning: Could not determine CPU count. Using 1 worker.")

        print("=" * 60)

        # Prepare arguments tuple for each worker call
        args_list = [
            (self.solver_class, self.objective_name, self.n, self.k,
             self.num_particles, self.T, self.dt, self.solver_params,
             self.objective_params, trial_id)
            for trial_id in range(self.trials)
        ]

        start_time_total = time.time()

        # --- Parallel Execution ---
        try:
            with Pool(num_workers) as pool:
                # Use tqdm to show progress over the imap_unordered iterator
                results_iterator = tqdm(
                    pool.imap_unordered(_run_single_trial_worker, args_list),
                    total=self.trials,
                    desc=f"Running {solver_name} Trials",
                    unit="trial"
                )
                self.results_list = list(results_iterator) # Collect all results

        except Exception as e:
            print(f"\nFATAL ERROR during multiprocessing pool execution: {e}")
            print("Ensure the C++ module and dependencies are correctly installed and accessible by worker processes.")
            # Optionally: Try running a single trial serially for easier debugging
            # print("Attempting single serial run for debugging...")
            # self.results_list = [_run_single_trial_worker(args_list[0])]


        end_time_total = time.time()
        self.total_runtime = end_time_total - start_time_total

        # --- Aggregate and Summarize Results ---
        self._aggregate_results()
        self._print_summary(solver_name)

    def _aggregate_results(self):
        """Processes the list of dictionaries returned by the workers."""
        if not self.results_list:
            print("Warning: No results collected from workers.")
            return

        # Filter out potentially failed trials (where results might be minimal/None)
        valid_results = [res for res in self.results_list if res and 'f_history' in res]

        if not valid_results:
             print("Warning: No valid results found after filtering worker outputs.")
             return

        self.all_f_M_dynamics = [res['f_history'] for res in valid_results]

        # Ensure histories have consistent length for aggregation (use first valid one as reference)
        ref_len = len(valid_results[0]['f_history'])
        self.all_f_M_dynamics = [hist if len(hist) == ref_len else np.full(ref_len, np.nan) for hist in self.all_f_M_dynamics]


        final_values = [hist[-1] for hist in self.all_f_M_dynamics if len(hist) > 0]
        self.asymptotic_values = np.array([f for f in final_values if np.isfinite(f)])

        # Set time points from the first successful result
        self.time_points = valid_results[0].get('time', np.linspace(0, self.T, ref_len))
        if len(self.time_points) != ref_len: # Adjust time points if mismatch
             self.time_points = np.linspace(0, self.T, ref_len)


        if self.asymptotic_values.size > 0:
            self.success_count = np.sum(self.asymptotic_values < self.success_tolerance)
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
            print(f"Success Rate (<{self.success_tolerance:.1e}): {self.success_count}/{self.trials} ({self.success_count/self.trials*100:.2f}%)")
            print(f"Mean Final f(M): {np.mean(self.asymptotic_values):.6e}")
            print(f"Std Dev Final f(M): {np.std(self.asymptotic_values):.6e}")
            print(f"Min Final f(M):  {np.min(self.asymptotic_values):.6e}")
            print(f"Max Final f(M):  {np.max(self.asymptotic_values):.6e}")
        else:
            print("Warning: No valid asymptotic data collected across all trials.")
        print("=" * 60)

    def get_results_for_plotting(self):
        """Returns the aggregated data needed for generating plots."""
        # Need objective instance metadata for plotting titles/labels
        # Recreate one instance here (this assumes objective_params don't change behavior significantly)
        plotting_objective_instance = None
        try:
             if self.objective_name == "Ackley":
                 plotting_objective_instance = cbo_module.AckleyObjective(self.n, self.k, **self.objective_params)
             elif self.objective_name == "WOPP":
                    plotting_objective_instance = cbo_module.WOPPObjective(self.n, self.k, **self.objective_params)
             elif self.objective_name == "QAP":
                    plotting_objective_instance = cbo_module.QAPObjective(self.n, self.k, **self.objective_params)
             # Add elif for WOPP, QAP...
        except Exception:
             # Fallback if objective creation fails for plotting
             class MockObjective:
                 def __init__(self, n, k): self.n, self.k = n, k
             plotting_objective_instance = MockObjective(self.n, self.k)


        return self.time_points, self.all_f_M_dynamics, self.asymptotic_values, plotting_objective_instance

# Note: This file now only defines the ExperimentRunner and its worker.
# The actual execution logic (`if __name__ == '__main__':`) should reside in
# files like `full_ackley_suite.py` or `wopp_experiment.py`.