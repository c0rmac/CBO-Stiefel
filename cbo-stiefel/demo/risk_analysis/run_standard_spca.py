import pandas as pd
import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler


def run_standard_sparse_pca(csv_file_path, k=5, alpha=1.0):
    """
    Loads asset returns, runs a standard (non-robust, local) Sparse PCA,
    and prints the resulting sparse factor matrix.

    Args:
        csv_file_path (str): Path to the asset_returns.csv file.
        k (int): The number of factors (principal components) to find.
        alpha (float): The sparsity-controlling parameter (like gamma).
                       Higher alpha = more sparse.
    """

    print("--- Running Standard Sparse PCA (sklearn) ---")

    # --- 1. Load and Prepare Data ---
    try:
        returns_df = pd.read_csv(csv_file_path)

        # Get asset tickers (column names)
        # Assumes the first column is 'Date'
        asset_tickers = list(returns_df.columns[1:])
        n = len(asset_tickers)

        # Get the numerical data, dropping the 'Date' column
        numerical_data = returns_df.iloc[:, 1:].to_numpy()

        print(f"Loaded {n} assets and {len(numerical_data)} observations.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- 2. Standardize the Data ---
    # PCA and Sparse PCA are sensitive to scale.
    # We must standardize to mean=0, variance=1.
    print("Standardizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numerical_data)

    # --- 3. Configure and Run Sparse PCA ---

    print(f"Fitting SparsePCA with k={k} and alpha={alpha}...")

    # n_jobs=-1 uses all available CPU cores
    # random_state=42 ensures reproducibility
    spca = SparsePCA(
        n_components=k,
        alpha=alpha,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model to the scaled data
    spca.fit(data_scaled)

    print("Fit complete.")

    # --- 4. Format and Display Results ---

    # scikit-learn stores components as (k, n)
    # We transpose (.T) to get the (n, k) format
    # that matches your Tessera project's output.
    M_sparse = spca.components_.T

    # Create column labels for the k factors
    factor_labels = [f'Factor_{i + 1}' for i in range(k)]

    # Create the labeled DataFrame
    factor_loadings_df = pd.DataFrame(
        M_sparse,
        index=asset_tickers,
        columns=factor_labels
    )

    # Set a display threshold to hide near-zero values
    pd.set_option('display.float_format', lambda x: f'{x: .4f}' if abs(x) > 1e-4 else ' .    ')

    print("\n" + "=" * 50)
    print("--- Standard Sparse PCA Factor Loadings (M) ---")
    print(factor_loadings_df)
    print("=" * 50)

    return factor_loadings_df


# --- Main execution ---
if __name__ == "__main__":
    # --- Parameters to Tune ---

    # This is the k-value you used in your experiment
    K_FACTORS = 5

    # This is the "gamma" for sklearn.
    # You MUST tune this value. Start here, and then
    # increase it to get more sparsity (more ' . ')
    # or decrease it to get denser factors.
    ALPHA_SPARSITY = 1.5

    run_standard_sparse_pca(
        csv_file_path="../asset_returns.csv",
        k=K_FACTORS,
        alpha=ALPHA_SPARSITY
    )