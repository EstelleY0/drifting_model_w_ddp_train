import torch
import torch.nn.functional as F


def compute_V(x, y_p, y_q, tau, mask_self=True):
    """
    Compute Drifting Field V

    x: Generated samples
    y_p: Positive samples from data distribution p
    y_q: Negative samples from generated distribution q
    tau: Temperature parameter
    """
    N = x.shape[0]
    N_pos = y_p.shape[0]
    N_neg = y_q.shape[0]

    # Calculate logits using pairwise L2 distances
    # logit = -||x - y|| / tau
    logit_p = -torch.cdist(x, y_p, p=2) / tau
    logit_q = -torch.cdist(x, y_q, p=2) / tau

    # Mask self-distances for generated samples
    # dist_neg += eye(N) * 1e6
    if mask_self and N == N_neg:
        logit_q.diagonal().fill_(-1e6)

    # Joint normalization using softmax along both dimensions
    # A = sqrt(softmax(logit, dim=-1) * softmax(logit, dim=-2))
    logit = torch.cat([logit_p, logit_q], dim=1)

    log_A_row = F.log_softmax(logit, dim=-1)
    log_A_col = F.log_softmax(logit, dim=-2)
    A = (0.5 * (log_A_row + log_A_col)).exp()

    # Split weight matrix back to positive and negative components
    A_p, A_q = A.split([N_pos, N_neg], dim=1)

    # Compute weights using cross-weighting strategy
    # W_pos = A_pos * A_neg.sum(dim=1), W_neg = A_neg * A_pos.sum(dim=1)
    W_p = A_p * A_q.sum(dim=1, keepdim=True)
    W_q = A_q * A_p.sum(dim=1, keepdim=True)

    # Calculate drifting field
    # V = (W_p @ y_p) - (W_q @ y_q) [cite: 151, 170, 565]
    V = (W_p @ y_p) - (W_q @ y_q)
    return V


def compute_V_multi_tau(x, y_p, y_q, tau_set=[0.02, 0.05, 0.2], mask_self=True, normalize_each=True):
    """
    Compute aggregated drifting field across multiple temperatures
    """
    V_total = torch.zeros_like(x)

    for tau_j in tau_set:
        # Compute V for each individual temperature scale
        V_tau = compute_V(x, y_p, y_q, tau_j, mask_self)

        if normalize_each:
            # Normalize field
            # V_tau / sqrt(E[||V_tau||^2]) [cite: 669, 676]
            V_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / V_norm

        V_total += V_tau

    return V_total


def normalize_phi(phi_x, S_j=None, mean=None, std=None):
    """
    Normalize features phi based on Section A.6 and Eq. 18 to Eq. 21
    Target average distance follows Eq. 21
    """
    C_j = phi_x.shape[1]
    target_dist = C_j ** 0.5

    with torch.no_grad():
        if mean is None:
            mean = phi_x.mean(dim=0, keepdim=True)
        if std is None:
            std = phi_x.std(dim=0, keepdim=True) + 1e-8

    # Standardize features
    phi_std = (phi_x - mean) / std

    # Calculate or apply normalization scale S_j
    if S_j is None:
        with torch.no_grad():
            n_sample = min(phi_x.shape[0], 256)
            subset = phi_std[:n_sample]

            # Pairwise distance for empirical mean estimation
            dists = torch.cdist(subset, subset, p=2)

            # Create mask for off-diagonal elements
            mask = ~torch.eye(n_sample, dtype=torch.bool, device=phi_x.device)
            avg_dist = dists[mask].mean()

            S_j = (target_dist / (avg_dist + 1e-8)).item()

    return phi_std * S_j, S_j, mean, std


def normalize_V(V, expected_V_norm=1.0):
    """
    Normalize drift field V
    V / lambda_j where lambda_j = sqrt(E[||V||^2 / C_j]) [cite: 666, 669]
    """
    current_var = torch.mean(V ** 2)
    lambda_j = (expected_V_norm / (current_var + 1e-8)) ** 0.5
    return V * lambda_j


def drift_step_2d(x, y_p, tau=0.1, eta=0.1):
    """
    Perform a single drift step for 2D samples
    x = x + eta * V [cite: 107, 216]
    """
    V = compute_V(x, y_p, x, tau, mask_self=True)
    return x + eta * V
