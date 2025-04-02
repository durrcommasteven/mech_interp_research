import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



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
                                 visualize: bool = False):
    """
    Performs LDA while accounting for token-specific biases.

    For each token, the class-conditional means (mu_0 and mu_1) are computed,
    and their average is used as the token-specific center. Each data point is
    then mean-centered by subtracting the average of its token's class means.
    The centered data across all tokens is combined, and LDA is performed to
    find a discriminative projection direction that is invariant to token biases.

    Additionally, the function outputs a confidence value and a magnitude associated 
    with the LDA projection vector. The unnormalized LDA vector (w_raw) is computed,
    its magnitude is used as a measure of the overall scale, and the confidence value
    is defined as the magnitude divided by the standard deviation of its components.

    Parameters:
        data_0 (dict[str, np.ndarray or torch.Tensor]): Class 0 token data.
        data_1 (dict[str, np.ndarray or torch.Tensor]): Class 1 token data.
        visualize (bool): If True, plot the 2D projection of the LDA result.

    Returns:
        w (np.ndarray): Normalized LDA projection vector.
        projected_0 (np.ndarray): Projected class 0 data.
        projected_1 (np.ndarray): Projected class 1 data.
        confidence (float): Confidence value of the LDA estimate.
        magnitude (float): Magnitude of the unnormalized LDA projection vector.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import torch
    except ImportError:
        torch = None

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
        if torch is not None:
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

    # Compute magnitude and normalize to obtain LDA vector
    magnitude = np.linalg.norm(w_raw)
    if magnitude == 0:
        return None
        raise ValueError("Computed LDA vector has zero magnitude.")
    
    w = w_raw / magnitude

    # Compute confidence as magnitude divided by the standard deviation of w_raw's components
    std_w = np.std(w_raw)
    epsilon = 1e-8  # Small constant to avoid division by zero
    confidence = magnitude / (std_w + epsilon)

    # Project data onto the LDA direction
    projected_0 = X0_all @ w
    projected_1 = X1_all @ w

    if visualize:
        # Create a second direction orthogonal to w for a 2D projection using SVD
        all_data = np.vstack([X0_all, X1_all])
        # Remove projection on w
        all_data_orth = all_data - (all_data @ w)[:, None] * w[None, :]
        u, s, vh = np.linalg.svd(all_data_orth, full_matrices=False)
        v = vh[0]
        v = v - (v @ w) * w  # Enforce orthogonality
        v = v / np.linalg.norm(v)

        W2D = np.vstack([w, v]).T
        X0_2d = X0_all @ W2D
        X1_2d = X1_all @ W2D

        plt.figure(figsize=(8, 6))
        plt.scatter(X0_2d[:, 0], X0_2d[:, 1], label='Class 0', alpha=0.7)
        plt.scatter(X1_2d[:, 0], X1_2d[:, 1], label='Class 1', alpha=0.7)
        plt.xlabel('LDA Direction')
        plt.ylabel('Orthogonal Direction')
        plt.title('Token-Normalized LDA Projection')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return w, projected_0, projected_1, confidence, magnitude
