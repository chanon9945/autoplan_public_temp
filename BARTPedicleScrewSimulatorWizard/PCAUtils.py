import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

def apply_pca(volume):
    pca = PCA(n_components=3)
    pca.fit(volume)
    coeff = pca.components_
    latent = pca.explained_variance_ratio_
    score = pca.transform(volume)
    return coeff,latent,score

def select_best_alignment_vector(principal_axes, target_axis=None):
    """
    Select the principal component vector that best aligns with the target axis.
    
    Parameters:
        principal_axes: numpy.ndarray of shape (3, 3) containing the PCA vectors as rows
                        (typically from pca.components_)
        target_axis: numpy.ndarray of shape (3,) specifying the target axis to align with
                     Default is [0, 1, 0] (posterior-anterior direction / Y-axis)
    
    Returns:
        tuple: (best_axis_index, best_axis_vector, alignment_score)
    """
    import numpy as np
    
    # Default target axis is the Y-axis (posterior-anterior direction)
    if target_axis is None:
        target_axis = np.array([0, 1, 0])
    
    # Normalize the target axis
    target_axis = target_axis / np.linalg.norm(target_axis)
    
    # Initialize variables to track the best alignment
    best_alignment = -1  # Will store the highest absolute dot product value
    best_idx = 0         # Will store the index of the best-aligned vector
    best_vector = None   # Will store the best-aligned vector itself
    
    # Check alignment of each principal axis
    for i in range(principal_axes.shape[0]):
        # Get the current axis
        axis = principal_axes[i]
        
        # Normalize the axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            continue  # Skip if the axis is essentially zero length
        
        axis = axis / axis_norm
        
        # Calculate the absolute dot product (alignment score)
        # Using absolute value because we care about alignment, not direction
        alignment = abs(np.dot(axis, target_axis))
        
        # Update if this is better than our current best
        if alignment > best_alignment:
            best_alignment = alignment
            best_idx = i
            best_vector = principal_axes[i]
    
    return best_idx, best_vector, best_alignment