import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

def mc_mean(transforms, resolution):
    """Calculate mean of transformation matrices"""
    try:
        # Handle different array dimensions
        if isinstance(transforms, (int, float)):
            return transforms
            
        if transforms.ndim == 3:
            sum_matrix = np.zeros_like(transforms[:, :, 0])
            for i in range(resolution):
                sum_matrix += transforms[:, :, i]
        elif transforms.ndim == 2:
            sum_matrix = np.zeros_like(transforms)
            for i in range(resolution):
                # Here we're assuming all matrices are the same
                sum_matrix += transforms
        elif transforms.ndim == 1:
            sum_matrix = np.zeros_like(transforms)
            for i in range(resolution):
                sum_matrix += transforms
        else:
            logger.error(f"Unexpected transform dimension: {transforms.ndim}")
            return np.zeros((3, 3))
            
        return sum_matrix / resolution
    except Exception as e:
        logger.error(f"Error in mc_mean: {str(e)}")
        if isinstance(transforms, np.ndarray):
            if transforms.ndim == 1:
                return np.zeros(transforms.shape)
            else:
                return np.zeros((3, 3))
        else:
            return 0.0

def m_correction(matrix):
    """Apply orthogonalization correction to rotation matrix"""
    try:
        # Handle the case where matrix might not be a full matrix
        if isinstance(matrix, (int, float)) or matrix.size == 0:
            return np.eye(3)
            
        # Use SVD for a more stable orthogonalization
        u, _, vh = np.linalg.svd(matrix, full_matrices=True)
        return u @ vh
    except Exception as e:
        logger.error(f"Error in m_correction: {str(e)}")
        return np.eye(3)

class HomogeneousTransform:
    def __init__(self, axis='x', angle=0, translate=None):
        """
        Create a homogeneous transformation matrix.
        
        Parameters:
            axis: Rotation axis ('x', 'y', or 'z')
            angle: Rotation angle in degrees
            translate: Translation vector (3,) or (3,1)
        """
        if translate is None:
            translate = np.zeros((3, 1))
        elif isinstance(translate, np.ndarray) and translate.ndim == 1:
            translate = translate.reshape(3, 1)
            
        try:
            # Create a 3x3 rotation matrix from Euler angles
            self.rotm = R.from_euler(axis, angle, degrees=True).as_matrix()
            
            # Append a row [0,0,0] to form a 4x3 matrix
            upper_part = np.concatenate((self.rotm, np.array([[0, 0, 0]])), axis=0)
            
            # Store the translation (ensure it is (3,1))
            self.translation = translate
            
            # Append [1] to the translation vector to form a (4,1) column
            temp_translation = np.concatenate((translate, np.array([[1]])), axis=0)
            
            # Concatenate horizontally to form a 4x4 homogeneous transformation matrix
            self.transform = np.concatenate((upper_part, temp_translation), axis=1)
        except Exception as e:
            logger.error(f"Error creating HomogeneousTransform: {str(e)}")
            # Create identity transform on error
            self.rotm = np.eye(3)
            self.translation = np.zeros((3, 1))
            self.transform = np.eye(4)
    
    def get_transform(self):
        """Return the 4x4 transformation matrix"""
        return self.transform

class Link:
    def __init__(self, axis, resolution, translate=None, limits=None):
        """
        Initialize a robot link.
        
        Parameters:
            axis: Rotation axis ('x', 'y', or 'z')
            resolution: Number of samples in joint space
            translate: Translation vector
            limits: Joint limits [min, max] in degrees
        """
        if translate is None:
            translate = np.zeros((3, 1))
        elif isinstance(translate, np.ndarray) and translate.ndim == 1:
            translate = translate.reshape(3, 1)
            
        if limits is None:
            limits = [-180, 180]
            
        try:
            # Preallocate a 4x4x(resolution) array for the transformation matrices
            self.transform = np.zeros((4, 4, resolution))
            self.joint_limit = np.linspace(limits[0], limits[1], resolution)
            
            for i in range(resolution):
                # Compose two transformations using matrix multiplication (@)
                t1 = HomogeneousTransform(axis, self.joint_limit[i]).get_transform()
                t2 = HomogeneousTransform(axis, 0, translate).get_transform()
                self.transform[:, :, i] = t1 @ t2
                
        except Exception as e:
            logger.error(f"Error initializing Link: {str(e)}")
            # Create default transform
            self.transform = np.zeros((4, 4, resolution))
            self.joint_limit = np.linspace(limits[0], limits[1], resolution)
            for i in range(resolution):
                self.transform[:, :, i] = np.eye(4)
    
    def get_transform(self, index):
        """Get transformation matrix at specified joint index"""
        if 0 <= index < self.transform.shape[2]:
            return self.transform[:, :, index]
        else:
            return np.eye(4)

class Robot:
    def __init__(self, resolution, reach):
        """
        Initialize a robot with two links.
        
        Parameters:
            resolution: Number of samples in joint space
            reach: Maximum reach distance
        """
        self.resolution = resolution
        self.reach = reach
        self.joint_1_limit = [-180, 180]
        self.joint_2_limit = [-90, 90]
        
        try:
            # Create two links
            self.link_1 = Link('y', resolution, limits=self.joint_1_limit)
            reach_vector = np.array([0, reach, 0]).reshape(3, 1)
            self.link_2 = Link('z', resolution, translate=reach_vector, limits=self.joint_2_limit)
        except Exception as e:
            logger.error(f"Error initializing Robot: {str(e)}")

    def get_mean_transforms(self):
        """Calculate mean transforms for both links"""
        try:
            # Initialize the mean matrices
            h1_mean = np.eye(4)
            h2_mean = np.eye(4)
            
            # Process rotation components
            h1_mean_r = np.zeros((3, 3))
            h2_mean_r = np.zeros((3, 3))
            
            # Process each transform separately to avoid dimension issues
            for i in range(self.resolution):
                # Extract rotation part (3x3 upper left)
                h1_mean_r += self.link_1.transform[0:3, 0:3, i]
                h2_mean_r += self.link_2.transform[0:3, 0:3, i]
            
            # Average the rotation matrices
            h1_mean_r = h1_mean_r / self.resolution
            h2_mean_r = h2_mean_r / self.resolution
            
            # Correct rotation matrices to be orthogonal
            h1_mean_r = m_correction(h1_mean_r)
            h2_mean_r = m_correction(h2_mean_r)
            
            # Initialize translation vectors
            h1_mean_t = np.zeros(3)
            h2_mean_t = np.zeros(3)
            
            # Process translation components element-wise
            for i in range(self.resolution):
                h1_mean_t[0] += self.link_1.transform[0, 3, i]
                h1_mean_t[1] += self.link_1.transform[1, 3, i]
                h1_mean_t[2] += self.link_1.transform[2, 3, i]
                
                h2_mean_t[0] += self.link_2.transform[0, 3, i]
                h2_mean_t[1] += self.link_2.transform[1, 3, i]
                h2_mean_t[2] += self.link_2.transform[2, 3, i]
            
            # Average the translation vectors
            h1_mean_t = h1_mean_t / self.resolution
            h2_mean_t = h2_mean_t / self.resolution
            
            # Assemble the final homogeneous transforms
            h1_mean[0:3, 0:3] = h1_mean_r
            h1_mean[0:3, 3] = h1_mean_t
            
            h2_mean[0:3, 0:3] = h2_mean_r
            h2_mean[0:3, 3] = h2_mean_t
            
            return h1_mean, h2_mean
            
        except Exception as e:
            logger.error(f"Error in get_mean_transforms: {str(e)}")
            # Return identity transforms on error
            return np.eye(4), np.eye(4)