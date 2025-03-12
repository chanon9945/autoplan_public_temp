import numpy as np
import scipy as sp
import vtk
from vtkmodules.util import numpy_support

imageMath = vtk.vtkImageMathematics()

def get_masked_centroid(volume, threshold=0):
    """
    Compute the centroid of nonzero voxels in a masked volume (vtkImageData).
    The centroid is calculated as the mean of the voxel coordinates (not intensity-weighted).
    
    Parameters:
      volume: vtkImageData containing the masked volume.
      threshold: value above which voxels are considered "on" (default 0).
    
    Returns:
      A list [x, y, z] representing the centroid in world coordinates, or None if no voxels exceed the threshold.
    """
    # Get volume properties
    dims = volume.GetDimensions()       # (nx, ny, nz)
    spacing = volume.GetSpacing()         # (sx, sy, sz)
    origin = volume.GetOrigin()           # (ox, oy, oz)
    
    # Convert the volume's scalar data to a NumPy array
    vtk_array = volume.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)
    # Reshape into a 3D array. VTK typically stores data in (z, y, x) order.
    np_array = np_array.reshape(dims[::-1])
    
    # Find indices where voxel values are above the threshold (nonzero in a binary mask)
    indices = np.argwhere(np_array > threshold)
    
    if indices.size == 0:
        return None  # No voxels found above the threshold.
    
    # Convert indices (z,y,x) to world coordinates (x,y,z):
    # x = origin[0] + index_x * spacing[0]
    # y = origin[1] + index_y * spacing[1]
    # z = origin[2] + index_z * spacing[2]
    # Note: indices[:,2] are x-indices, indices[:,1] are y-indices, indices[:,0] are z-indices.
    world_coords = np.empty((indices.shape[0], 3), dtype=float)
    world_coords[:, 0] = origin[0] + indices[:, 2] * spacing[0]
    world_coords[:, 1] = origin[1] + indices[:, 1] * spacing[1]
    world_coords[:, 2] = origin[2] + indices[:, 0] * spacing[2]
    
    # Compute the geometric centroid as the mean of all points.
    centroid = np.mean(world_coords, axis=0)
    return centroid

# Example usage:
# volume is your vtkImageData of the masked volume.
# centroid = get_masked_centroid(volume, threshold=0)
# print("Centroid:", centroid)


def maskedVolumeToPointCloud(maskedVolume, threshold=0):
    """
    Convert a masked vtkImageData (volume) into a point cloud (vtkPolyData).
    Only voxels with values greater than threshold are kept.
    
    Parameters:
      maskedVolume: vtkImageData representing the masked volume.
      threshold: Value above which voxels are considered part of the point cloud.
      
    Returns:
      vtkPolyData representing the point cloud.
    """
    # Get volume dimensions, spacing, and origin
    dims = maskedVolume.GetDimensions()       # (x, y, z)
    spacing = maskedVolume.GetSpacing()         # (sx, sy, sz)
    origin = maskedVolume.GetOrigin()           # (ox, oy, oz)
    
    # Convert the VTK array to a NumPy array
    vtk_array = maskedVolume.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)
    
    # Reshape the flat array into 3D (note: VTK typically stores in z, y, x order)
    np_array = np_array.reshape(dims[::-1])  # shape: (z, y, x)
    
    # Find indices where voxel values exceed the threshold
    # (Assuming binary mask: voxels > 0 are "on")
    indices = np.argwhere(np_array > threshold)
    
    # Convert voxel indices (z, y, x) to world coordinates (x, y, z)
    points = []
    for idx in indices:
        # idx is in (z, y, x) order
        x = origin[0] + idx[2] * spacing[0]
        y = origin[1] + idx[1] * spacing[1]
        z = origin[2] + idx[0] * spacing[2]
        points.append([x, y, z])
    points = np.array(points)
    
    # Create vtkPoints and insert the world coordinates
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)
    
    # Create a vtkPolyData to hold the point cloud
    pointCloud = vtk.vtkPolyData()
    pointCloud.SetPoints(vtk_points)
    
    return pointCloud

# Example usage:
# Assume 'maskedVolume' is your vtkImageData that you got after multiplying your volume by the binary mask.
# For a binary mask, threshold=0 means we keep all nonzero voxels.
# pointCloudPolyData = maskedVolumeToPointCloud(maskedVolume, threshold=0)

def get_masked_centroid(volume, threshold=0):
    """
    Compute the centroid of nonzero voxels in a masked volume (vtkImageData).
    The centroid is calculated as the mean of the voxel coordinates (not intensity-weighted).
    
    Parameters:
      volume: vtkImageData containing the masked volume.
      threshold: value above which voxels are considered "on" (default 0).
    
    Returns:
      A list [x, y, z] representing the centroid in world coordinates, or None if no voxels exceed the threshold.
    """
    # Get volume properties
    dims = volume.GetDimensions()       # (nx, ny, nz)
    spacing = volume.GetSpacing()         # (sx, sy, sz)
    origin = volume.GetOrigin()           # (ox, oy, oz)
    
    # Convert the volume's scalar data to a NumPy array
    vtk_array = volume.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)
    # Reshape into a 3D array. VTK typically stores data in (z, y, x) order.
    np_array = np_array.reshape(dims[::-1])
    
    # Find indices where voxel values are above the threshold (nonzero in a binary mask)
    indices = np.argwhere(np_array > threshold)
    
    if indices.size == 0:
        return None  # No voxels found above the threshold.
    
    # Convert indices (z,y,x) to world coordinates (x,y,z):
    # x = origin[0] + index_x * spacing[0]
    # y = origin[1] + index_y * spacing[1]
    # z = origin[2] + index_z * spacing[2]
    # Note: indices[:,2] are x-indices, indices[:,1] are y-indices, indices[:,0] are z-indices.
    world_coords = np.empty((indices.shape[0], 3), dtype=float)
    world_coords[:, 0] = origin[0] + indices[:, 2] * spacing[0]
    world_coords[:, 1] = origin[1] + indices[:, 1] * spacing[1]
    world_coords[:, 2] = origin[2] + indices[:, 0] * spacing[2]
    
    # Compute the geometric centroid as the mean of all points.
    centroid = np.mean(world_coords, axis=0)
    return centroid

# Example usage:
# volume is your vtkImageData of the masked volume.
# centroid = get_masked_centroid(volume, threshold=0)
# print("Centroid:", centroid)

def centroid_shift(coordinateX, coordinateY, coordinateZ, maskIn):
    """
    Recursively finds a shifted centroid along the Y-axis for the given mask.
    
    If coordinateY is at the top of the image (or first index) or if the mask at
    (coordinateX, coordinateY, coordinateZ) is 0, returns the current coordinates.
    Otherwise, it decrements coordinateY until one of those conditions is met.
    
    Parameters:
        coordinateX (int): The X index.
        coordinateY (int): The Y index (0-indexed).
        coordinateZ (int): The Z index.
        maskIn (np.ndarray): A 3D NumPy array representing the mask.
        
    Returns:
        (shiftedX, shiftedY) (tuple): The adjusted X and Y indices.
    """
    # Base case: if we are at the first row (index 0) then return the current position.
    if coordinateY <= 0:
        return coordinateX, coordinateY
    
    # If the mask at the given coordinate is 0, return the current coordinates.
    if maskIn[coordinateX, coordinateY, coordinateZ] == 0:
        return coordinateX, coordinateY
    else:
        # Otherwise, recursively decrement the Y coordinate.
        return centroid_shift(coordinateX, coordinateY - 1, coordinateZ, maskIn)

class Vertebra:
    def __init__(self,mask,volume):
        self.mask = mask
        self.volume = volume
        imageMath.SetOperationToMultiply()
        imageMath.SetInputData(0, volume)
        imageMath.SetInputData(1, mask)
        imageMath.Update()
        self.maskedVolume = imageMath.GetOutput()
        self.pt_cloud = maskedVolumeToPointCloud(self.maskedVolume)
        self.centroid = get_masked_centroid(self.maskedVolume)