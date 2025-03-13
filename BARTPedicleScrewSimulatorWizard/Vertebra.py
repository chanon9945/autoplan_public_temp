import numpy as np
import scipy as sp
import vtk
from vtkmodules.util import numpy_support
from PCAUtils import *

imageMath = vtk.vtkImageMathematics()

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
    
import numpy as np

def dfs(image, image_size, i, j, shape, stackX, stackY):
    # Check if (i,j) is out-of-bounds.
    if i < 0 or i >= image_size[0] or j < 0 or j >= image_size[1]:
        return shape, stackX, stackY
    
    # If this pixel has already been filled, return.
    if shape[i, j] != 0:
        return shape, stackX, stackY
    
    # (Optional) You could also check a condition on 'image' here if needed.
    # For now, we'll fill regardless of the image value.
    
    # Mark the current pixel as filled.
    shape[i, j] = 1
    
    # Add the current coordinates to the stack lists.
    stackX.append(i)
    stackY.append(j)
    
    # Recursively visit 4-connected neighbors.
    shape, stackX, stackY = dfs(image, image_size, i + 1, j, shape, stackX, stackY)
    shape, stackX, stackY = dfs(image, image_size, i - 1, j, shape, stackX, stackY)
    shape, stackX, stackY = dfs(image, image_size, i, j + 1, shape, stackX, stackY)
    shape, stackX, stackY = dfs(image, image_size, i, j - 1, shape, stackX, stackY)
    
    return shape, stackX, stackY

def dfs_iterative(image, image_size, start_i, start_j):
    shape = np.zeros((image_size[0], image_size[1]), dtype=image.dtype)
    stack_x = []
    stack_y = []
    
    # Use a stack for DFS
    to_visit = [(start_i, start_j)]
    
    while to_visit:
        i, j = to_visit.pop()
        
        # Skip if out of bounds or already visited
        if i < 0 or i >= image_size[0] or j < 0 or j >= image_size[1]:
            continue
        if shape[i, j] != 0:
            continue
        
        # Mark as visited
        shape[i, j] = 1
        stack_x.append(i)
        stack_y.append(j)
        
        # Add neighbors to stack
        to_visit.append((i+1, j))
        to_visit.append((i-1, j))
        to_visit.append((i, j+1))
        to_visit.append((i, j-1))
    
    return shape, stack_x, stack_y

def floodfill(image, i, j):
    """
    Flood fill starting from the seed coordinate (i, j) on a 2D NumPy array.
    
    Parameters:
      image (np.ndarray): 2D array representing the image.
      i (int): Row index of the seed.
      j (int): Column index of the seed.
    
    Returns:
      filled_shape (np.ndarray): A 2D array (same size as image) with the filled region marked (1's).
      stackX (list): List of row indices visited.
      stackY (list): List of column indices visited.
    """
    image_size = image.shape  # (rows, cols)
    # Create a 2D array (shape) initialized to zeros.
    shape = np.zeros((image_size[0], image_size[1]), dtype=image.dtype)
    # Initialize stacks as lists with the seed coordinate.
    stackX = [i]
    stackY = [j]
    
    filled_shape, stackX, stackY = dfs_iterative(image, image_size, i, j, shape, stackX, stackY)
    return filled_shape, stackX, stackY

def cut_pedicle(volIn, yMax, yMin, bufferFront=15, bufferEnd=1):
    """
    Zero out parts of the volume along the second dimension.
    
    Parameters:
      volIn: 3D NumPy array (e.g. shape: (rows, columns, slices))
      yMax: Maximum column index (Python 0-indexed) for the region of interest.
      yMin: Minimum column index (Python 0-indexed) for the region of interest.
      bufferFront: Number of extra columns at the front to cut.
      bufferEnd: Number of extra columns at the end to cut.
      
    Returns:
      Modified volume with specified columns zeroed out.
    """
    # Set the front part (columns 0 to yMin+bufferFront, inclusive) to 0.
    volIn[:, 0:yMin+bufferFront+1, :] = 0
    # Set the end part (columns from yMax-bufferEnd to the end) to 0.
    volIn[:, yMax-bufferEnd:, :] = 0
    return volIn

def pedicle_roi_cut(vol, roiPointX, centroidX):
    """
    Cut the volume along the x-dimension based on the roiPointX and centroidX.
    
    If roiPointX > centroidX, then all slices from the beginning up to centroidX
    (i.e. vol[0:centroidX, :, :]) are set to zero.
    Otherwise, all slices from centroidX to the end are set to zero.
    
    Parameters:
      vol (np.ndarray): 3D volume (e.g., shape: (x, y, z))
      roiPointX (int): The x-coordinate of the ROI point.
      centroidX (int): The x-coordinate of the centroid.
    
    Returns:
      np.ndarray: The modified volume with the specified region zeroed out.
    """
    if roiPointX > centroidX:
        vol[:centroidX, :, :] = 0
    else:
        vol[centroidX:, :, :] = 0
    return vol

import numpy as np
from scipy.ndimage import binary_erosion

def extract_surface_voxels(volume):
    """
    Extract the surface voxels from a binary volume.
    
    This function computes the binary erosion of the input volume and then subtracts
    the eroded volume from the original volume to obtain the boundary voxels.
    
    Parameters:
      volume (np.ndarray): A binary 3D (or 2D) NumPy array.
      
    Returns:
      np.ndarray: A binary array of the same shape with True at surface voxels.
    """
    # Perform binary erosion on the volume.
    eroded = binary_erosion(volume)
    
    # The surface voxels are those that are in the original volume but not in the eroded volume.
    surface_voxels = volume & (~eroded)
    return surface_voxels

def center(shape):
    """
    Computes the centroid of the nonzero region of a 2D array using 0-indexing.
    
    In MATLAB, ndgrid(1:M,1:N) produces 1-indexed coordinates; here we use 0-indexed
    coordinates by creating grids over the range 0 to M-1 and 0 to N-1.
    
    Parameters:
      shape (np.ndarray): A 2D NumPy array.
      
    Returns:
      center (tuple): (centerX, centerY) as a tuple of integers (0-indexed).
      centerX (int): The row coordinate of the centroid.
      centerY (int): The column coordinate of the centroid.
    """
    # Create a binary mask of nonzero values.
    nonZeroMask = shape > 0
    
    # Create coordinate grids using 0-indexing.
    xnew, ynew = np.mgrid[0:shape.shape[0], 0:shape.shape[1]]
    
    # Compute the mean of the coordinates where the mask is True.
    centerX = int(np.round(np.mean(xnew[nonZeroMask])))
    centerY = int(np.round(np.mean(ynew[nonZeroMask])))
    
    center_coord = (centerX, centerY)
    return center_coord, centerX, centerY

class Vertebra:
    def __init__(self,mask,volume,insertion_point):
        self.mask = mask
        self.volume = volume
        self.insertion_point = insertion_point
        imageMath.SetOperationToMultiply()
        imageMath.SetInputData(0, volume)
        imageMath.SetInputData(1, mask)
        imageMath.Update()
        self.maskedVolume = imageMath.GetOutput()
        self.centroid = get_masked_centroid(self.maskedVolume)

        shifted_x, shifted_y = centroid_shift(self.centroid[0],self.centroid[1],self.centroid[2],self.mask)
        self.spinal_canal, stack_x, stack_y = floodfill(self.mask[:,:,self.centroid[2]],shifted_x,shifted_y)
        canal_max_y = np.max(stack_y)
        canal_min_y = np.min(stack_y)
        canal_center, canal_center_x, canal_center_y = center(self.spinal_canal)
        self.pedicle = cut_pedicle(self.maskedVolume,canal_max_y,canal_min_y)
        self.pedicle_mask = cut_pedicle(self.mask,canal_max_y,canal_min_y)

        self.pedicle_ROI = pedicle_roi_cut(self.pedicle,self.insertion_point[0],self.centroid[0])
        self.pedicle_ROI_mask = pedicle_roi_cut(self.pedicle_mask,self.insertion_point[0],self.centroid[0])

        self.pedicle_point_cloud = maskedVolumeToPointCloud(self.pedicle_ROI_mask)

        coeff,latent,score = apply_pca(self.pedicle_point_cloud)
        self.pedicle_center_point = np.mean(np.array(self.pedicle_point_cloud), axis=0)
        scalingFactor = np.sqrt(latent)*2
        self.pcaVectors = coeff * scalingFactor

        self.surface = extract_surface_voxels(self.maskedVolume)
        self.point_cloud = maskedVolumeToPointCloud(self.surface)
