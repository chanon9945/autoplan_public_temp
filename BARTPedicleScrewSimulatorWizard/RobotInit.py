import numpy as np
from scipy.spatial.transform import Rotation as R

def mc_mean(transforms, resolution):
    sum_matrix = np.zeros_like(transforms[:, :, 0])
    for i in range(resolution):
        sum_matrix += transforms[:, :, i]
    return sum_matrix / resolution

def m_correction(matrix):
    return matrix @ np.linalg.inv(np.linalg.inv(matrix.T @ matrix)**0.5)

class HomogeneousTransform:
    def __init__(self, axis='x', angle=0, translate=np.array([0, 0, 0]).reshape(3, 1)):
        # Create a 3x3 rotation matrix from Euler angles.
        self.rotm = R.from_euler(axis, angle, degrees=True).as_matrix()
        
        # Append a row [0,0,0] to form a 4x3 matrix.
        upper_part = np.concatenate((self.rotm, np.array([[0, 0, 0]])), axis=0)
        
        # Store the translation (ensure it is (3,1)).
        self.translation = translate
        
        # Append [1] to the translation vector to form a (4,1) column.
        temp_translation = np.concatenate((translate, np.array([[1]])), axis=0)
        
        # Concatenate horizontally to form a 4x4 homogeneous transformation matrix.
        self.transform = np.concatenate((upper_part, temp_translation), axis=1)
    
    def get_transform(self):
        return self.transform

class Link:
    def __init__(self, axis, resolution, translate=np.array([0, 0, 0]).reshape(3, 1), limits=[-180, 180]):
        # Preallocate a 4x4x(resolution) array for the transformation matrices.
        self.transform = np.zeros((4, 4, resolution))
        self.joint_limit = np.linspace(limits[0], limits[1], resolution)
        for i in range(resolution):
            # Compose two transformations using matrix multiplication (@).
            t1 = HomogeneousTransform(axis, self.joint_limit[i]).get_transform()
            t2 = HomogeneousTransform(axis, 0, translate).get_transform()
            self.transform[:, :, i] = t1 @ t2
    
    def get_transform(self,index):
        return self.transform[:,:,index]

class Robot:
    def __init__(self, resolution, reach):
        self.resolution = resolution
        self.reach = reach
        self.joint_1_limit = [-180, 180]
        self.joint_2_limit = [-90, 90]
        self.link_1 = Link('y', resolution, limits=self.joint_1_limit)
        self.link_2 = Link('z', resolution, translate=np.array([0, reach, 0]).reshape(3, 1), limits=self.joint_2_limit)

    def get_mean_transforms(self):
        # Mean of rotation matrices
        h1_mean_m = mc_mean(self.link_1.transform[0:3, 0:3, :], self.resolution)
        h2_mean_m = mc_mean(self.link_2.transform[0:3, 0:3, :], self.resolution)
        
        # Format correction
        h1_mean_r = m_correction(h1_mean_m)
        h2_mean_r = m_correction(h2_mean_m)
        
        # Mean of translation
        h1_mean_t = mc_mean(self.link_1.transform[0:3, 3, :], self.resolution).reshape(3, 1)
        h2_mean_t = mc_mean(self.link_2.transform[0:3, 3, :], self.resolution).reshape(3, 1)
        
        # Create homogeneous transforms
        h1_mean = np.zeros((4, 4))
        h2_mean = np.zeros((4, 4))
        
        h1_mean[0:3, 0:3] = h1_mean_r
        h1_mean[0:3, 3] = h1_mean_t.flatten()
        h1_mean[3, 3] = 1.0
        
        h2_mean[0:3, 0:3] = h2_mean_r
        h2_mean[0:3, 3] = h2_mean_t.flatten()
        h2_mean[3, 3] = 1.0
        
        return h1_mean, h2_mean
