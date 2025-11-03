import numpy as np
import pandas as pd  # Optional, but helpful for viewing labeled data

# Define the path where the matrix X was saved
SAVE_DIR = "asset_returns_experiment"
file_path = f"saves/{SAVE_DIR}/best_minimizer_X.npy"

try:
    # 1. Load the NPY file
    X_minimizer = np.load(file_path)

    # 2. Display the matrix properties
    print(f"--- Loaded Minimizer Matrix X ---")
    print(f"File: {file_path}")
    print(f"Shape (n x k): {X_minimizer.shape}")
    print(f"Number of Features (n): {X_minimizer.shape[0]}")
    print(f"Number of Robust Factors (k): {X_minimizer.shape[1]}")
    print("-" * 30)

    # 3. Display the top-left corner of the matrix
    # This shows the weights of the first few assets on the first few factors.
    print("Top-left corner (first 10 rows, all factors):")
    print(X_minimizer)

    # 4. (Optional) Count sparsity (if gamma > 0 was used)
    # A small non-zero threshold is often used due to floating-point numbers.
    sparsity_threshold = 1e-6
    non_zero_count = np.sum(np.abs(X_minimizer) > sparsity_threshold)
    total_elements = X_minimizer.size

    print("-" * 30)
    print(f"Sparsity Check (Tolerance: {sparsity_threshold}):")
    print(f"Total elements: {total_elements}")
    print(f"Non-zero elements: {non_zero_count}")
    print(f"Sparsity: {100 * (1 - non_zero_count / total_elements):.2f}%")

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    print("Please ensure the experiment ran successfully and created the 'saves/asset_returns_experiment' directory.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")