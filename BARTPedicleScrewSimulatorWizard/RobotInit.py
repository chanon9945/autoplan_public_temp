import numpy as np
from scipy.spatial.transform import Rotation as R

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
