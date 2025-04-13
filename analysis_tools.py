import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import os
import yaml


def lda_two_class(X1, X2, reg_param=0):
    """
    Perform Linear Discriminant Analysis (LDA) for two classes.
    
    Parameters:
    X1 (numpy.ndarray): Data matrix for class 1, shape (B1, D)
    X2 (numpy.ndarray): Data matrix for class 2, shape (B2, D)
    reg_param (float): Regularization parameter to add to the within-class scatter matrix
    
    Returns:
    w (numpy.ndarray): LDA projection vector, shape (D,)
    projected_X1 (numpy.ndarray): Projected class 1 data, shape (B1,)
    projected_X2 (numpy.ndarray): Projected class 2 data, shape (B2,)
    """
    n1, d = X1.shape
    n2, d2 = X2.shape
    assert d == d2, "Feature dimensions must match"
    
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    
    # Vectorized computation of within-class scatter matrices
    S1 = (X1 - mu1).T @ (X1 - mu1)
    S2 = (X2 - mu2).T @ (X2 - mu2)
    Sw = S1 + S2
    
    # Optional regularization for numerical stability
    if reg_param:
        Sw += reg_param * np.eye(d)
    
    # Compute projection vector
    diff = mu1 - mu2
    w = np.linalg.pinv(Sw) @ diff
    
    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        raise ValueError("Degenerate solution: the projection vector is zero.")
    w = w / norm_w
    
    projected_X1 = X1 @ w
    projected_X2 = X2 @ w
    
    return w, projected_X1, projected_X2


def optimal_2d_projection(X1, X2, reg_param=0):
    """
    Produce an optimal 2D projection for two classes by combining
    the discriminative LDA direction with an additional variance direction.
    
    Parameters:
      X1 (numpy.ndarray): Data matrix for class 1, shape (N1, D)
      X2 (numpy.ndarray): Data matrix for class 2, shape (N2, D)
      reg_param (float): Regularization parameter for within-class scatter (default 0)
      
    Returns:
      projection_matrix (numpy.ndarray): Matrix with shape (D, 2) whose columns
                                           are the projection directions.
      proj_X1 (numpy.ndarray): Projected class 1 data in 2D, shape (N1, 2)
      proj_X2 (numpy.ndarray): Projected class 2 data in 2D, shape (N2, 2)
    """
    # Check dimensions
    N1, d = X1.shape
    N2, d2 = X2.shape
    assert d == d2, "Feature dimensions must match"
    
    # Compute class means
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    
    # Compute within-class scatter matrices (vectorized)
    S1 = (X1 - mu1).T @ (X1 - mu1)
    S2 = (X2 - mu2).T @ (X2 - mu2)
    Sw = S1 + S2
    
    # Optionally add regularization for numerical stability
    if reg_param:
        Sw += reg_param * np.eye(d)
    
    # Compute LDA direction (maximizes between-class variance)
    diff = mu1 - mu2
    w = np.linalg.pinv(Sw) @ diff
    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        raise ValueError("Degenerate solution: LDA projection vector is zero.")
    w = w / norm_w  # normalize LDA vector
    
    # Combine data for PCA
    X = np.vstack((X1, X2))
    
    # Compute PCA on the entire dataset to capture overall variance
    pca = PCA(n_components=2)
    pca.fit(X)
    comps = pca.components_  # shape (2, d)
    
    # Choose the PCA component that is most orthogonal to the LDA direction
    dot_products = np.abs(comps @ w)
    idx = np.argmin(dot_products)
    v = comps[idx]
    
    # Ensure v is orthogonal to w (project out any component along w)
    v = v - (v @ w) * w
    v = v / np.linalg.norm(v)
    
    # Form the 2D projection matrix (each column is a direction)
    projection_matrix = np.vstack([w, v]).T  # shape (d, 2)
    
    # Project each class onto the 2D subspace
    proj_X1 = X1 @ projection_matrix
    proj_X2 = X2 @ projection_matrix
    
    return projection_matrix, proj_X1, proj_X2



def perform_token_normalized_lda(data_0: dict[str, "np.ndarray | torch.Tensor"],
                                 data_1: dict[str, "np.ndarray | torch.Tensor"],
                                 visualize: bool = False,
                                 save_location: str|None = None):
    """
    Performs LDA while accounting for token-specific biases.

    For each token, the class-conditional means (mu_0 and mu_1) are computed,
    and their average (unweighted -- meaning 
        mu_mean = (mu_0 + mu_1)/2
    )
    is used as the token-specific center. Each data point is
    then mean-centered by subtracting the average of its token's class means.
    mu_normed0 = mu_0 - mu_mean = (mu_0-mu_1)/2
    mu_normed1 = mu_1 - mu_mean = (mu_1-mu_0)/2
    The centered data across all tokens is combined, and LDA is performed to
    find a discriminative projection direction that is invariant to token biases.

    Additionally, the function outputs a confidence value and a magnitude associated 
    with the LDA projection vector. The unnormalized LDA vector (w_raw) is computed,
    its magnitude is used as a measure of the overall scale, and the confidence value
    is defined as the magnitude divided by the standard deviation of its components.

    if visualize, plot a projection of the token-normalized data_0 and data_1.
    We do this by taking the x axis to be the w vector
    And for the y vector, we find the axis orthogonal to w which captures the 
    most variance. 

    Parameters:
        data_0 (dict[str, np.ndarray or torch.Tensor]): Class 0 token data.
        data_1 (dict[str, np.ndarray or torch.Tensor]): Class 1 token data.
        visualize (bool): If True, plot the 2D projection of the LDA result.

    Returns:
        w (np.ndarray): Normalized LDA projection vector.
        projected_0 (np.ndarray): Projected class 0 data (1D projection along w).
        projected_1 (np.ndarray): Projected class 1 data (1D projection along w).
        magnitude (float): Magnitude of the unnormalized LDA projection vector.
    """

    # Keep only tokens present in both classes
    common_tokens = set(data_0.keys()) & set(data_1.keys())
    if not common_tokens:
        raise ValueError("No common tokens found between the two classes.")
    
    X0_centered = []
    X1_centered = []

    for token in common_tokens:
        X0 = data_0[token]
        X1 = data_1[token]

        # Convert torch tensors to NumPy arrays if necessary
        if 'torch' in globals() and torch is not None:
            if isinstance(X0, torch.Tensor):
                X0 = X0.detach().cpu().to(torch.float32).numpy()
            if isinstance(X1, torch.Tensor):
                X1 = X1.detach().cpu().to(torch.float32).numpy()

        mu0_token = np.mean(X0, axis=0)
        mu1_token = np.mean(X1, axis=0)
        mu_token = 0.5 * (mu0_token + mu1_token)

        X0c = X0 - mu_token
        X1c = X1 - mu_token

        X0_centered.append(X0c)
        X1_centered.append(X1c)

    # Combine all centered data
    X0_all = np.vstack(X0_centered)
    X1_all = np.vstack(X1_centered)

    # Compute overall means for each class
    mu0_all = np.mean(X0_all, axis=0)
    mu1_all = np.mean(X1_all, axis=0)

    # Compute within-class scatter matrices
    S0 = (X0_all - mu0_all).T @ (X0_all - mu0_all)
    S1 = (X1_all - mu1_all).T @ (X1_all - mu1_all)
    Sw = S0 + S1

    # Compute the unnormalized LDA direction
    diff = mu0_all - mu1_all
    w_raw = np.linalg.pinv(Sw) @ diff

    diff_mag = np.linalg.norm(diff)

    # Compute magnitude and normalize to obtain LDA vector
    magnitude = np.linalg.norm(w_raw)
    if magnitude == 0:
        raise ValueError("Computed LDA vector has zero magnitude.")
    
    w = w_raw / magnitude

    # Compute 1D projections (along the LDA direction) to return.
    projected_0 = X0_all @ w
    projected_1 = X1_all @ w

    # Visualization block
    if visualize:
        # Compute the 2D transformation: first axis is w, second axis captures max variance orthogonal to w.
        X_all = np.vstack([X0_all, X1_all])
        # Project onto w (1D) and subtract to get residuals orthogonal to w.
        proj_on_w = (X_all @ w).reshape(-1, 1) * w.reshape(1, -1)
        residual = X_all - proj_on_w
        
        # Compute the dominant orthogonal direction via SVD.
        # (Since residual is orthogonal to w, the first right-singular vector is our y-axis.)
        _, _, Vt = np.linalg.svd(residual, full_matrices=False)
        y_axis = Vt[0]
        
        # Build the transformation matrix T whose columns are w and y_axis.
        T = np.column_stack((w, y_axis))
        
        # Compute 2D projections for visualization.
        proj0_2d = X0_all @ T
        proj1_2d = X1_all @ T

        # Create scatter plot.
        plt.figure()
        plt.scatter(proj0_2d[:, 0], proj0_2d[:, 1], label='Class 0', alpha=0.5)
        plt.scatter(proj1_2d[:, 0], proj1_2d[:, 1], label='Class 1', alpha=0.5)

        # Project the overall class means into the 2D space.
        #mean0_2d = mu0_all.dot(T)
        #mean1_2d = mu1_all.dot(T)

        # Plot an arrow representing the un-normalized LDA vector connecting the class means.
        #arrow_dx = mean0_2d[0] - mean1_2d[0]
        #arrow_dy = mean0_2d[1] - mean1_2d[1]
        #plt.arrow(mean1_2d[0], mean1_2d[1], arrow_dx, arrow_dy,
        #          width=0.01, color='red', length_includes_head=True,
        #          label='LDA direction')
        
        plt.xlabel("Projection on LDA direction")
        plt.ylabel("Projection on orthogonal axis")
        plt.title("LDA Projection of Token-Normalized Data")
        plt.legend()
        plt.tight_layout()

        # If a save location is provided, ensure the directory exists and save the figure.
        if save_location is not None:
            save_dir = os.path.dirname(save_location)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_location)
        plt.show()

    return w, projected_0, projected_1, diff_mag


def evaluate_token_distinguishability(dict_0: dict[str, "np.ndarray | torch.Tensor"],
                                      dict_1: dict[str, "np.ndarray | torch.Tensor"],
                                      w: np.ndarray):
    """
    Evaluates token-specific metrics that quantify how well each token distinguishes
    between two classes. For every token in the intersection of dict_0 and dict_1,
    the function computes the following:
    
        - mu_diff = (mu_1 - mu_0) / 2, where mu_0 and mu_1 are the means of the token data in dict_0 and dict_1.
        - Cosine similarity between mu_diff and the LDA vector w.
        - Strength: the projection of mu_diff onto w (i.e. np.dot(mu_diff, w)).
        - Magnitude: the Euclidean norm of mu_diff.
        - Confidence: defined as the absolute strength divided by the pooled standard deviation of
                      the projections of the token's data onto w.
                      
    Note:
        Other metrics that may be informative include a t-statistic for the difference of means 
        (taking into account the sample sizes and variances) or Cohen's d effect size.
        
    Parameters:
        dict_0 (dict[str, np.ndarray or torch.Tensor]): Class 0 token data.
        dict_1 (dict[str, np.ndarray or torch.Tensor]): Class 1 token data.
        w (np.ndarray): LDA projection vector (assumed to be normalized).
        
    Returns:
        results (dict): A dictionary mapping each token to its metrics, with keys:
            - "cosine_similarity": Alignment between mu_diff and w.
            - "strength": Projection of mu_diff onto w.
            - "magnitude": Norm of mu_diff.
            - "confidence": Strength relative to the pooled standard deviation of projections.
            - "sigma0": Standard deviation of class 0 projections.
            - "sigma1": Standard deviation of class 1 projections.
            - "pooled_sigma": Average of sigma0 and sigma1.
    """
    results = {}
    common_tokens = set(dict_0.keys()) & set(dict_1.keys())
    
    for token in common_tokens:
        X0 = dict_0[token]
        X1 = dict_1[token]
        
        # Convert torch tensors to numpy arrays if necessary.
        if 'torch' in globals() and torch is not None:
            if isinstance(X0, torch.Tensor):
                X0 = X0.detach().cpu().to(torch.float32).numpy()
            if isinstance(X1, torch.Tensor):
                X1 = X1.detach().cpu().to(torch.float32).numpy()
        
        # Compute token-specific means.
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        mu_diff = (mu1 - mu0) / 2.0
        
        # Cosine similarity between mu_diff and w.
        norm_mu_diff = np.linalg.norm(mu_diff)
        if norm_mu_diff > 0:
            cosine_similarity = np.dot(mu_diff, w) / norm_mu_diff
        else:
            cosine_similarity = 0.0
        
        # Strength: projection of mu_diff onto w.
        strength = np.dot(mu_diff, w)
        
        # Magnitude of mu_diff.
        magnitude = norm_mu_diff
        
        # Project each token's points onto w.
        proj0 = X0 @ w
        proj1 = X1 @ w
        
        # Compute standard deviations along w.
        sigma0 = np.std(proj0)
        sigma1 = np.std(proj1)
        pooled_sigma = (sigma0 + sigma1) / 2.0
        
        # Confidence: signal-to-noise ratio for the token.
        if pooled_sigma > 0:
            confidence = np.abs(strength) / pooled_sigma
        else:
            confidence = np.nan
        
        results[str(token)] = {
            "cosine_similarity": cosine_similarity,
            "strength": strength,
            "magnitude": magnitude,
            "confidence": confidence,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "pooled_sigma": pooled_sigma
        }
        
    return results


def plot_token_normalized_2d(data_0: dict[str, "np.ndarray | torch.Tensor"],
                             data_1: dict[str, "np.ndarray | torch.Tensor"],
                             save_location: str | None = None):
    """
    Plots a 2D projection of token-normalized data from data_0 and data_1.

    For each token found in both dictionaries, the token-specific data is normalized by
    subtracting its center (the average of the class means for that token). The normalized
    data from all tokens is combined, and Singular Value Decomposition (SVD) is used to
    extract the two directions capturing maximum variation. The data from each class is
    then projected into this 2D space and plotted with different colors.

    Parameters:
        data_0 (dict[str, np.ndarray or torch.Tensor]): Class 0 token data.
        data_1 (dict[str, np.ndarray or torch.Tensor]): Class 1 token data.
        save_location (str or None): File path to save the plot. If provided, the directory
                                     will be created if it does not exist. If None, the plot
                                     is simply shown.
    """
    # Identify tokens present in both classes.
    common_tokens = set(data_0.keys()) & set(data_1.keys())
    if not common_tokens:
        raise ValueError("No common tokens found between the two classes.")
    
    X0_centered = []
    X1_centered = []

    # Normalize each token's data by subtracting the token-specific center.
    for token in common_tokens:
        X0 = data_0[token]
        X1 = data_1[token]

        # Convert torch tensors to NumPy arrays if necessary.
        if 'torch' in globals() and torch is not None:
            if isinstance(X0, torch.Tensor):
                X0 = X0.detach().cpu().to(torch.float32).numpy()
            if isinstance(X1, torch.Tensor):
                X1 = X1.detach().cpu().to(torch.float32).numpy()

        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        mu_token = 0.5 * (mu0 + mu1)

        X0_centered.append(X0 - mu_token)
        X1_centered.append(X1 - mu_token)

    # Combine all token-normalized data.
    X0_all = np.vstack(X0_centered)
    X1_all = np.vstack(X1_centered)
    X_all = np.vstack([X0_all, X1_all])

    # Use SVD to find the two directions expressing maximum variation.
    U, S, Vt = np.linalg.svd(X_all, full_matrices=False)
    # The top two right-singular vectors (rows of Vt) represent the principal directions.
    T = Vt[:2, :].T  # Transformation matrix (d x 2)

    proj0_2d = X0_all @ T
    proj1_2d = X1_all @ T

    # Create scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(proj0_2d[:, 0], proj0_2d[:, 1], label='Class 0', alpha=0.6, color='blue')
    plt.scatter(proj1_2d[:, 0], proj1_2d[:, 1], label='Class 1', alpha=0.6, color='red')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D SVD Projection of Token-Normalized Data")
    plt.legend()
    plt.tight_layout()

    # Save the plot if a save location is provided.
    if save_location is not None:
        save_dir = os.path.dirname(save_location)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_location)
    
    plt.show()


def plot_lda_1d_histograms(projected_0: np.ndarray,
                            projected_1: np.ndarray,
                            bins: int = 50,
                            alpha: float = 0.5,
                            save_location: str | None = None):
    """
    Plots overlapping 1D histograms of LDA-projected class 0 and class 1 data.
    
    Parameters:
        projected_0 (np.ndarray): LDA projections of class 0.
        projected_1 (np.ndarray): LDA projections of class 1.
        bins (int): Number of histogram bins.
        alpha (float): Transparency for overlapping histograms.
        save_location (str or None): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(projected_0, bins=bins, alpha=alpha, label="Class 0", color='blue', density=True)
    plt.hist(projected_1, bins=bins, alpha=alpha, label="Class 1", color='red', density=True)

    plt.xlabel("Projection onto LDA direction")
    plt.ylabel("Density")
    plt.title("1D LDA Projection Histogram")
    plt.legend()
    plt.tight_layout()

    if save_location is not None:
        import os
        save_dir = os.path.dirname(save_location)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_location)

    plt.show()


def perform_token_normalized_analysis_suite(data_0: dict[str, "np.ndarray | torch.Tensor"],
                                              data_1: dict[str, "np.ndarray | torch.Tensor"],
                                              save_location: str = 'outputs',
                                              descriptor: str | None = None):
    """
    Performs a set of simple analyses while accounting for token-specific biases.

    For each token, the class-conditional means (mu_0 and mu_1) are computed,
    and their average (unweighted -- meaning 
        mu_mean = (mu_0 + mu_1)/2
    )
    is used as the token-specific center. Each data point is
    then mean-centered by subtracting the average of its token's class means.
    mu_normed0 = mu_0 - mu_mean
    mu_normed1 = mu_1 - mu_mean
    The centered data across all tokens is combined, and we perform a 
    suite of analyses.

    The analyses performed include:
      - LDA computation with a corresponding projection plot.
      - 1D histogram plots of the LDA projections.
      - A 2D SVD projection of the token-normalized data.
      - Analysis of separation along the LDA vector (t-statistics and Cohen's d).
      - Evaluation of token distinguishability (cosine similarity, strength, etc.).

    The numerical outputs of all analyses are aggregated and saved as a YAML file
    in save_location, and each plot is saved with a descriptive filename.
    
    For the arrays, we save them using NumPys binary format (.npy), which preserves
    datatypes (e.g., np.float32) without conversion.
    
    Parameters:
        data_0 (dict[str, np.ndarray or torch.Tensor]): Class 0 token data.
        data_1 (dict[str, np.ndarray or torch.Tensor]): Class 1 token data.
        save_location (str): Directory path for outputs and plots.
        descriptor (str or None): An optional descriptor to include in plot filenames.

    Returns:
        results (dict): A dictionary mapping analysis names to their outputs.
    """
    # Ensure the save_location directory exists.
    if not os.path.exists(save_location):
        os.makedirs(save_location, exist_ok=True)
    
    # Sanitize descriptor for filenames.
    desc_str = ""
    if descriptor is not None:
        desc_str = descriptor.strip().replace(" ", "_") + "_"
    
    # Define file paths for the plots.
    lda_projection_plot = os.path.join(save_location, f"{desc_str}lda_projection.png")
    lda_1d_hist_plot    = os.path.join(save_location, f"{desc_str}lda_1d_histogram.png")
    token_norm_2d_plot  = os.path.join(save_location, f"{desc_str}token_normalized_2d.png")
    
    # 1. Perform token-normalized LDA analysis with visualization.
    #    This returns the LDA projection vector, 1D projections, and its magnitude.
    print("doing LDA")
    w, projected_0, projected_1, diff_mag = perform_token_normalized_lda(
        data_0=data_0,
        data_1=data_1,
        visualize=True,
        save_location=lda_projection_plot
    )
    
    # 2. Plot 1D histograms of the LDA projections.
    print("plotting 1d hists")
    plot_lda_1d_histograms(
        projected_0=projected_0,
        projected_1=projected_1,
        bins=50,
        alpha=0.5,
        save_location=lda_1d_hist_plot
    )
    
    # 3. Plot the 2D projection of the token-normalized data using SVD.
    print("plotting 2d projection")
    plot_token_normalized_2d(
        data_0=data_0,
        data_1=data_1,
        save_location=token_norm_2d_plot
    )
    
    # 5. Evaluate token distinguishability metrics (e.g., cosine similarity, strength).
    print("How well does each token difference align with LDA w")
    token_distinguishability = evaluate_token_distinguishability(
        dict_0=data_0,
        dict_1=data_1,
        w=w
    )
    
    # Save the arrays in their native binary format.
    w_file = os.path.join(save_location, f"{desc_str}w_vector.npy")
    proj0_file = os.path.join(save_location, f"{desc_str}projected_0.npy")
    proj1_file = os.path.join(save_location, f"{desc_str}projected_1.npy")
    
    np.save(w_file, w)
    np.save(proj0_file, projected_0)
    np.save(proj1_file, projected_1)
    
    # Aggregate numerical and file-path results into a dictionary.
    results = {
        "lda": {
            "w_file": w_file,
            "projected_0_file": proj0_file,
            "projected_1_file": proj1_file,
            "diff_mag": diff_mag,
        },
        "token_distinguishability": token_distinguishability,
        "plots": {
            "lda_projection": lda_projection_plot,
            "lda_1d_histogram": lda_1d_hist_plot,
            "token_normalized_2d": token_norm_2d_plot
        }
    }
    
    # Save the results as a YAML file.
    yaml_file_path = os.path.join(save_location, f"{desc_str}analysis_results.yaml")
    with open(yaml_file_path, "w") as f:
        yaml.dump(results, f)
    
    return results


def load_token_normalized_analysis_results(save_location: str, descriptor: str | None = None):
    """
    Loads the analysis results saved by perform_token_normalized_analysis_suite.

    It loads the YAML file (which contains metadata and file paths for the NumPy arrays),
    then loads the arrays in their native format. The returned dictionary maps descriptive
    keys to the loaded values.

    Parameters:
        save_location (str): Directory path where the results were saved.
        descriptor (str or None): The descriptor used during saving (if any).

    Returns:
        results (dict): A dictionary with analysis results. The 'lda' entry includes the loaded
                        NumPy arrays for the LDA vector and projections.
    """
    # Sanitize descriptor for filenames.
    desc_str = ""
    if descriptor is not None:
        desc_str = descriptor.strip().replace(" ", "_") + "_"
    
    yaml_file_path = os.path.join(save_location, f"{desc_str}analysis_results.yaml")
    
    # Load the YAML file.
    with open(yaml_file_path, "r") as f:
        results = yaml.safe_load(f)
    
    # Load the NumPy arrays from their respective files.
    w = np.load(results["lda"]["w_file"])
    projected_0 = np.load(results["lda"]["projected_0_file"])
    projected_1 = np.load(results["lda"]["projected_1_file"])
    
    # Insert the loaded arrays into the results dictionary.
    results["lda"]["w"] = w
    results["lda"]["projected_0"] = projected_0
    results["lda"]["projected_1"] = projected_1
    
    return results


