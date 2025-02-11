import numpy as np
import cvxpy as cp
import heapq
from numpy.fft import fft2, ifft2, fftfreq

# -----------------------------------------------
# 1. Goldstein’s Branch-Cut Unwrapping (Simplified)
# -----------------------------------------------
def goldstein_branch_cut_unwrap(wrapped_phase):
    """
    Simplified Goldstein Branch-Cut Phase Unwrapping.
    
    This implementation:
      - Computes a residue map from the wrapped phase.
      - Marks pixels with high residue as branch cuts.
      - Uses a flood-fill (stack) approach to unwrap the phase while avoiding branch cuts.
    
    Note: This is a very basic implementation that may fail in complex scenarios.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    def compute_residues(phase):
        rows, cols = phase.shape
        residues = np.zeros_like(phase)
        # Loop over interior pixels (avoid boundaries)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                d1 = np.angle(np.exp(1j * (phase[i, j+1] - phase[i, j])))
                d2 = np.angle(np.exp(1j * (phase[i+1, j+1] - phase[i, j+1])))
                d3 = np.angle(np.exp(1j * (phase[i+1, j] - phase[i+1, j+1])))
                d4 = np.angle(np.exp(1j * (phase[i, j] - phase[i+1, j])))
                residue = d1 + d2 + d3 + d4
                residues[i, j] = residue / (2 * np.pi)
        return residues

    residues = compute_residues(wrapped_phase)
    # Create branch cuts: mark pixels where |residue| is larger than a threshold.
    branch_cut = np.abs(residues) > 0.5  # Threshold can be tuned

    rows, cols = wrapped_phase.shape
    unwrapped = np.copy(wrapped_phase)
    visited = np.zeros_like(wrapped_phase, dtype=bool)
    # Start unwrapping from the top-left pixel (assumed not on a branch cut)
    start = (0, 0)
    visited[start] = True
    stack = [start]

    while stack:
        i, j = stack.pop()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if visited[ni, nj] or branch_cut[ni, nj]:
                continue
            # Compute the phase difference and adjust it to be in [-pi, pi]
            diff = wrapped_phase[ni, nj] - wrapped_phase[i, j]
            diff = np.angle(np.exp(1j * diff))
            unwrapped[ni, nj] = unwrapped[i, j] + diff
            visited[ni, nj] = True
            stack.append((ni, nj))
    return unwrapped

# -----------------------------------------------
# 2. Least-Squares Phase Unwrapping using FFT
# -----------------------------------------------
def ls_unwrap_phase(wrapped_phase):
    """
    Least-Squares Phase Unwrapping using an FFT-based Poisson solver.
    
    The method computes finite differences in x and y, forms a divergence,
    and then solves the Poisson equation in the Fourier domain.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    M, N = wrapped_phase.shape

    # Compute wrapped finite differences along x and y directions.
    dx = np.zeros_like(wrapped_phase)
    dy = np.zeros_like(wrapped_phase)
    dx[:, :-1] = np.angle(np.exp(1j * (wrapped_phase[:, 1:] - wrapped_phase[:, :-1])))
    dy[:-1, :] = np.angle(np.exp(1j * (wrapped_phase[1:, :] - wrapped_phase[:-1, :])))

    # Compute divergence of the gradient differences.
    div = np.zeros_like(wrapped_phase)
    # For x-direction:
    div[:, 0] = dx[:, 0]
    div[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    div[:, -1] = -dx[:, -2]
    # For y-direction:
    div[0, :] += dy[0, :]
    div[1:-1, :] += dy[1:-1, :] - dy[:-2, :]
    div[-1, :] += -dy[-2, :]

    # Solve the Poisson equation using FFT.
    k1 = fftfreq(M).reshape(-1, 1)
    k2 = fftfreq(N).reshape(1, -1)
    # Discrete Laplacian eigenvalues (using cosine formulation)
    laplacian = (2 * np.cos(2 * np.pi * k1) - 2) + (2 * np.cos(2 * np.pi * k2) - 2)
    laplacian[0, 0] = 1  # Avoid division by zero for the DC term
    div_fft = fft2(div)
    unwrapped_fft = div_fft / laplacian
    unwrapped = np.real(ifft2(unwrapped_fft))
    # Remove arbitrary constant offset
    unwrapped -= unwrapped[0, 0]
    return unwrapped

# -----------------------------------------------
# 3. Quality-Guided Phase Unwrapping
# -----------------------------------------------
def quality_guided_unwrap(wrapped_phase):
    """
    Quality-Guided Phase Unwrapping.
    
    A simple implementation that computes a quality map based on the local gradient
    (the lower the gradient, the higher the quality), then unwraps the phase starting
    from the pixel with the highest quality, propagating to neighbors in a greedy fashion.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    rows, cols = wrapped_phase.shape
    # Compute a simple quality map: higher quality for lower gradient magnitude.
    grad_x = np.gradient(wrapped_phase, axis=1)
    grad_y = np.gradient(wrapped_phase, axis=0)
    quality = 1.0 / (np.abs(grad_x) + np.abs(grad_y) + 1e-6)
    
    unwrapped = np.full_like(wrapped_phase, np.nan)
    # Priority queue: use negative quality to simulate a max-heap.
    heap = []
    start_idx = np.unravel_index(np.argmax(quality), wrapped_phase.shape)
    unwrapped[start_idx] = wrapped_phase[start_idx]
    visited = np.zeros_like(wrapped_phase, dtype=bool)
    visited[start_idx] = True
    heapq.heappush(heap, (-quality[start_idx], start_idx))
    
    while heap:
        neg_q, (i, j) = heapq.heappop(heap)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if visited[ni, nj]:
                continue
            diff = wrapped_phase[ni, nj] - wrapped_phase[i, j]
            diff = np.angle(np.exp(1j * diff))
            unwrapped[ni, nj] = unwrapped[i, j] + diff
            visited[ni, nj] = True
            heapq.heappush(heap, (-quality[ni, nj], (ni, nj)))
    return unwrapped

# -----------------------------------------------
# 4. Minimum Lp-Norm Phase Unwrapping (Using cvxpy)
# -----------------------------------------------
def min_lp_unwrap(wrapped_phase, p=1):
    """
    Minimum Lp-Norm Phase Unwrapping using convex optimization.
    
    For p=2, this is equivalent to a least-squares formulation.
    For p=1 (or other values), the method minimizes the sum of the Lp norms
    of the differences between finite differences of the unwrapped phase and
    the wrapped phase differences.
    
    Parameters:
        wrapped_phase (np.ndarray): 2D array of wrapped phase (radians).
        p (float): The norm to be minimized (default is 1).
    
    Returns:
        np.ndarray: Unwrapped phase.
    """
    rows, cols = wrapped_phase.shape
    # Define the optimization variable.
    u = cp.Variable((rows, cols))
    
    # Finite differences for u (unwrapped phase) in x and y directions.
    dx = u[:, 1:] - u[:, :-1]
    dy = u[1:, :] - u[:-1, :]
    
    # Compute wrapped differences of the measured phase.
    wrapped_dx = np.angle(np.exp(1j * (wrapped_phase[:, 1:] - wrapped_phase[:, :-1])))
    wrapped_dy = np.angle(np.exp(1j * (wrapped_phase[1:, :] - wrapped_phase[:-1, :])))
    
    # Objective: minimize the sum of the Lp norms of the differences.
    objective = cp.Minimize(cp.sum(cp.abs(dx - wrapped_dx)**p) + cp.sum(cp.abs(dy - wrapped_dy)**p))
    # Constraint to fix the global phase (remove ambiguity).
    constraints = [u[0, 0] == wrapped_phase[0, 0]]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return u.value


# # -----------------------------------------------
# # Example Usage and Visualization
# # -----------------------------------------------

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Create a synthetic wrapped phase for demonstration.
#     x = np.linspace(0, 4 * np.pi, 100)
#     y = np.linspace(0, 4 * np.pi, 100)
#     X, Y = np.meshgrid(x, y)
#     true_phase = X + Y  # A smooth ramp.
#     wrapped_phase = np.angle(np.exp(1j * true_phase))
    
#     # Apply the different unwrapping algorithms.
#     unwrapped_goldstein = goldstein_branch_cut_unwrap(wrapped_phase)
#     unwrapped_ls = ls_unwrap_phase(wrapped_phase)
#     unwrapped_quality = quality_guided_unwrap(wrapped_phase)
#     unwrapped_min_lp = min_lp_unwrap(wrapped_phase, p=1)
    
#     # Plot the results for comparison.
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     im0 = axes[0, 0].imshow(wrapped_phase, cmap='jet')
#     axes[0, 0].set_title("Wrapped Phase")
#     plt.colorbar(im0, ax=axes[0, 0])
    
#     im1 = axes[0, 1].imshow(unwrapped_goldstein, cmap='jet')
#     axes[0, 1].set_title("Goldstein Branch-Cut")
#     plt.colorbar(im1, ax=axes[0, 1])
    
#     im2 = axes[0, 2].imshow(unwrapped_ls, cmap='jet')
#     axes[0, 2].set_title("Least-Squares Unwrap")
#     plt.colorbar(im2, ax=axes[0, 2])
    
#     im3 = axes[1, 0].imshow(unwrapped_quality, cmap='jet')
#     axes[1, 0].set_title("Quality-Guided Unwrap")
#     plt.colorbar(im3, ax=axes[1, 0])
    
#     im4 = axes[1, 1].imshow(unwrapped_min_lp, cmap='jet')
#     axes[1, 1].set_title("Minimum Lp Unwrap (p=1)")
#     plt.colorbar(im4, ax=axes[1, 1])
    
#     # Hide the unused subplot.
#     axes[1, 2].axis('off')
    
#     for ax in axes.flat:
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.show()
