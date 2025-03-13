import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca
import logging

class Vertebra:
    def __init__(self, mask, volume, insertion_point):
        """
        Initialize a Vertebra object for pedicle screw trajectory planning.
        
        Parameters:
            mask: vtkImageData representing the binary mask of the vertebra
            volume: vtkImageData representing the CT intensity values
            insertion_point: [x, y, z] coordinates of the surgeon-specified insertion point
        """
        self.mask = mask
        self.volume = volume
        self.insertion_point = np.array(insertion_point)
        
        # Set up logging for debugging
        self.logger = logging.getLogger(__name__)
        
        try:
            # Convert VTK volumes to NumPy arrays for processing
            self.mask_array = self._vtk_to_numpy_3d(mask)
            self.volume_array = self._vtk_to_numpy_3d(volume)
            
            # Multiply volume by mask to get masked volume
            self.masked_volume_array = self.volume_array * self.mask_array
            self.maskedVolume = self._numpy_to_vtk_3d(self.masked_volume_array, self.volume)
            
            # Calculate centroid of masked volume
            self.centroid = self._get_masked_centroid(self.masked_volume_array)
            if self.centroid is None:
                raise ValueError("Failed to compute centroid - masked volume may be empty")
            
            # Process spinal canal
            centroid_slice_z = int(self.centroid[2])
            mask_slice = self.mask_array[:, :, centroid_slice_z].copy()
            
            # Find a point inside the spinal canal
            shifted_x, shifted_y = self._centroid_shift(
                int(self.centroid[0]), 
                int(self.centroid[1]),
                centroid_slice_z, 
                self.mask_array
            )
            
            # Floodfill to find spinal canal
            self.spinal_canal, stack_x, stack_y = self._floodfill(
                mask_slice, 
                shifted_x, 
                shifted_y
            )
            
            # Find canal boundaries
            if len(stack_y) > 0:
                canal_max_y = np.max(stack_y)
                canal_min_y = np.min(stack_y)
            else:
                # Fallback if spinal canal not found
                canal_max_y = int(self.centroid[1]) 
                canal_min_y = int(self.centroid[1]) - 20
            
            # Cut pedicle using canal boundaries
            self.pedicle_array = self._cut_pedicle(
                self.masked_volume_array.copy(), 
                canal_max_y, 
                canal_min_y, 
                15, 1
            )
            
            self.pedicle_mask_array = self._cut_pedicle(
                self.mask_array.copy(), 
                canal_max_y, 
                canal_min_y, 
                15, 1
            )
            
            # Convert processed arrays back to VTK objects
            self.pedicle = self._numpy_to_vtk_3d(self.pedicle_array, self.maskedVolume)
            self.pedicle_mask = self._numpy_to_vtk_3d(self.pedicle_mask_array, self.mask)
            
            # Cut pedicle based on ROI point
            self.pedicle_roi_array = self._pedicle_roi_cut(
                self.pedicle_array.copy(),
                self.insertion_point[0],
                self.centroid[0]
            )
            
            self.pedicle_roi_mask_array = self._pedicle_roi_cut(
                self.pedicle_mask_array.copy(),
                self.insertion_point[0],
                self.centroid[0]
            )
            
            # Convert to VTK
            self.pedicle_roi = self._numpy_to_vtk_3d(self.pedicle_roi_array, self.maskedVolume)
            self.pedicle_roi_mask = self._numpy_to_vtk_3d(self.pedicle_roi_mask_array, self.mask)
            
            # Create point cloud and run PCA
            self.pedicle_point_cloud = self._masked_volume_to_point_cloud(self.pedicle_roi_mask)
            
            # Initialize default PCA vectors (identity matrix)
            self.pcaVectors = np.eye(3)
            
            if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
                # Convert VTK point cloud to numpy array for PCA
                points_array = self._vtk_points_to_numpy(self.pedicle_point_cloud.GetPoints())
                
                if points_array.shape[0] >= 3:  # Need at least 3 points for meaningful PCA
                    try:
                        coeff, latent, score = apply_pca(points_array)
                        self.pedicle_center_point = np.mean(points_array, axis=0)
                        scaling_factor = np.sqrt(latent) * 2
                        self.pcaVectors = coeff * scaling_factor[:, np.newaxis]
                    except Exception as e:
                        self.logger.error(f"PCA failed: {str(e)}")
            
            # Extract surface voxels
            self.surface_array = self._extract_surface_voxels(self.mask_array)
            self.surface = self._numpy_to_vtk_3d(self.surface_array, self.mask)
            self.point_cloud = self._masked_volume_to_point_cloud(self.surface)
            
        except Exception as e:
            self.logger.error(f"Vertebra initialization error: {str(e)}")
            raise

    def _vtk_to_numpy_3d(self, vtk_image):
        """Convert vtkImageData to 3D numpy array, with safety checks"""
        if not vtk_image:
            return np.zeros((1, 1, 1))
        
        dims = vtk_image.GetDimensions()
        point_data = vtk_image.GetPointData()
        if not point_data:
            return np.zeros(dims[::-1])
            
        scalar_data = point_data.GetScalars()
        if not scalar_data:
            return np.zeros(dims[::-1])
            
        numpy_data = numpy_support.vtk_to_numpy(scalar_data)
        
        # Check for zero dimensions to avoid reshape errors
        if 0 in dims:
            return np.zeros((1, 1, 1))
            
        # VTK stores data in format (z, y, x)
        try:
            return numpy_data.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)
        except ValueError:
            # If reshape fails, return zeros with correct dimensions
            self.logger.warning(f"Failed to reshape VTK data with dims {dims}")
            return np.zeros(dims[::-1])

    def _numpy_to_vtk_3d(self, numpy_array, reference_vtk=None):
        """Convert 3D numpy array to vtkImageData, with error handling"""
        if numpy_array is None or numpy_array.size == 0:
            self.logger.warning("Empty array passed to _numpy_to_vtk_3d")
            return vtk.vtkImageData()
            
        if reference_vtk is None:
            # Create new vtkImageData if no reference is provided
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(numpy_array.shape)
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
        else:
            # Clone properties from reference vtkImageData
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(reference_vtk.GetDimensions())
            vtk_image.SetSpacing(reference_vtk.GetSpacing())
            vtk_image.SetOrigin(reference_vtk.GetOrigin())
            
            # Copy direction matrix if available
            if hasattr(reference_vtk, 'GetDirectionMatrix'):
                direction_matrix = reference_vtk.GetDirectionMatrix()
                if direction_matrix:
                    vtk_image.SetDirectionMatrix(direction_matrix)
        
        try:    
            # Transpose array to VTK's expected format (x,y,z) -> (z,y,x)
            numpy_array_t = numpy_array.transpose(2, 1, 0).copy()
            flat_array = numpy_array_t.flatten()
            
            # Create VTK array and set it as the image data
            vtk_array = numpy_support.numpy_to_vtk(
                flat_array, 
                deep=True,
                array_type=vtk.VTK_FLOAT
            )
            vtk_image.GetPointData().SetScalars(vtk_array)
            
            return vtk_image
        except Exception as e:
            self.logger.error(f"Error in _numpy_to_vtk_3d: {str(e)}")
            return vtk.vtkImageData()

    def _get_masked_centroid(self, masked_volume_array, threshold=0):
        """
        Compute the centroid of nonzero voxels in a masked volume.
        Returns [x, y, z] coordinates in physical space.
        """
        try:
            # Find indices where the value is above threshold
            indices = np.where(masked_volume_array > threshold)
            
            if len(indices[0]) == 0:
                return None
                
            # Calculate centroid in index space
            centroid_i = np.mean(indices[0])
            centroid_j = np.mean(indices[1])
            centroid_k = np.mean(indices[2])
            
            # Get spacing and origin from the volume
            if hasattr(self, 'volume') and self.volume:
                spacing = self.volume.GetSpacing()
                origin = self.volume.GetOrigin()
                
                # Convert to physical coordinates
                centroid_x = origin[0] + spacing[0] * centroid_i
                centroid_y = origin[1] + spacing[1] * centroid_j
                centroid_z = origin[2] + spacing[2] * centroid_k
                
                return [centroid_x, centroid_y, centroid_z]
            else:
                # Return index coordinates if volume info not available
                return [centroid_i, centroid_j, centroid_k]
                
        except Exception as e:
            self.logger.error(f"Error in _get_masked_centroid: {str(e)}")
            return None

    def _centroid_shift(self, coord_x, coord_y, coord_z, mask_array, max_iterations=100):
        """
        Find a point inside the spinal canal by shifting posteriorly.
        Python implementation of centroidShift.m with iteration limit to prevent stack overflow.
        """
        # Use iterative approach instead of recursion to prevent stack overflow
        iteration = 0
        current_x, current_y = coord_x, coord_y
        
        while iteration < max_iterations:
            # Base case: we're at the edge or found an empty voxel
            if current_y <= 0:
                return current_x, current_y
                
            try:
                if mask_array[current_x, current_y, coord_z] == 0:
                    return current_x, current_y
            except IndexError:
                # Handle out-of-bounds access
                return max(0, min(current_x, mask_array.shape[0]-1)), max(0, min(current_y, mask_array.shape[1]-1))
                
            # Move backwards along Y
            current_y -= 1
            iteration += 1
            
        # If we hit max iterations, return current position
        return current_x, current_y

    def _floodfill(self, image, i, j):
        """
        Floodfill to find connected region.
        Python implementation of floodfill.m using iterative approach.
        """
        image_size = image.shape
        shape = np.zeros(image_size, dtype=image.dtype)
        
        # Validate start coordinates
        i = max(0, min(i, image_size[0]-1))
        j = max(0, min(j, image_size[1]-1))
        
        # Starting position must be empty
        if image[i, j] != 0:
            return shape, np.array([]), np.array([])
        
        # Use stack-based approach for flood fill
        stack = [(i, j)]
        stack_x = []
        stack_y = []
        
        # Define direction vectors for 4-connected neighbors
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while stack:
            x, y = stack.pop()
            
            # Skip if outside bounds or already filled or not empty in original
            if (x < 0 or y < 0 or x >= image_size[0] or y >= image_size[1] or
                    shape[x, y] != 0 or image[x, y] != 0):
                continue
            
            # Mark this pixel
            shape[x, y] = 1
            stack_x.append(x)
            stack_y.append(y)
            
            # Add neighbors to stack
            for dx, dy in directions:
                stack.append((x + dx, y + dy))
        
        return shape, np.array(stack_x), np.array(stack_y)

    def _cut_pedicle(self, vol_in, y_max, y_min, buffer_front=15, buffer_end=1):
        """
        Cut the pedicle using the spinal canal boundaries.
        Python implementation of cutPedicle.m with bounds checking.
        """
        # Ensure valid indices
        shape = vol_in.shape
        y_min_idx = max(0, min(int(y_min + buffer_front + 1), shape[1]-1))
        y_max_idx = max(0, min(int(y_max - buffer_end), shape[1]-1))
        
        # Zero out regions
        if y_min_idx > 0:
            vol_in[:, 0:y_min_idx, :] = 0
        if y_max_idx < shape[1]-1:
            vol_in[:, y_max_idx:, :] = 0
            
        return vol_in

    def _pedicle_roi_cut(self, vol, roi_point_x, centroid_x):
        """
        Cut the pedicle ROI based on the insertion point.
        Python implementation of pedicleROICut.m with bounds checking.
        """
        centroid_idx = int(centroid_x)
        shape = vol.shape
        
        # Ensure valid indices
        centroid_idx = max(0, min(centroid_idx, shape[0]-1))
        
        if roi_point_x > centroid_x:
            if centroid_idx > 0:
                vol[0:centroid_idx, :, :] = 0
        else:
            if centroid_idx < shape[0]:
                vol[centroid_idx:, :, :] = 0
                
        return vol

    def _extract_surface_voxels(self, volume):
        """
        Extract surface voxels from a binary volume.
        Python implementation of extractSurfaceVoxels.m using binary erosion.
        """
        from scipy.ndimage import binary_erosion
        
        # Perform binary erosion
        eroded = binary_erosion(volume > 0)
        
        # Surface voxels are in original but not in eroded
        surface_voxels = np.logical_and(volume > 0, np.logical_not(eroded))
        return surface_voxels.astype(volume.dtype)

    def _masked_volume_to_point_cloud(self, masked_volume, threshold=0):
        """
        Convert a masked volume to a VTK point cloud.
        """
        try:
            # Convert to numpy if it's a VTK object
            if isinstance(masked_volume, vtk.vtkImageData):
                numpy_array = self._vtk_to_numpy_3d(masked_volume)
            else:
                numpy_array = masked_volume
                
            # Get volume dimensions, spacing, and origin
            if isinstance(masked_volume, vtk.vtkImageData):
                dims = masked_volume.GetDimensions()
                spacing = masked_volume.GetSpacing()
                origin = masked_volume.GetOrigin()
            else:
                dims = numpy_array.shape
                spacing = (1.0, 1.0, 1.0)
                origin = (0.0, 0.0, 0.0)
            
            # Find indices of voxels above threshold
            indices = np.where(numpy_array > threshold)
            
            if len(indices[0]) == 0:
                # Return empty point cloud
                points = vtk.vtkPoints()
                point_cloud = vtk.vtkPolyData()
                point_cloud.SetPoints(points)
                return point_cloud
            
            # Create points in physical coordinates
            points = vtk.vtkPoints()
            # Pre-allocate for efficiency
            points.SetNumberOfPoints(len(indices[0]))
            
            for i in range(len(indices[0])):
                x = origin[0] + indices[0][i] * spacing[0]
                y = origin[1] + indices[1][i] * spacing[1]
                z = origin[2] + indices[2][i] * spacing[2]
                points.SetPoint(i, x, y, z)
            
            # Create polydata
            point_cloud = vtk.vtkPolyData()
            point_cloud.SetPoints(points)
            
            # Add vertex cells (improves rendering)
            vertices = vtk.vtkCellArray()
            for i in range(points.GetNumberOfPoints()):
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(i)
                
            point_cloud.SetVerts(vertices)
            
            return point_cloud
            
        except Exception as e:
            self.logger.error(f"Error in _masked_volume_to_point_cloud: {str(e)}")
            # Return an empty point cloud
            points = vtk.vtkPoints()
            point_cloud = vtk.vtkPolyData()
            point_cloud.SetPoints(points)
            return point_cloud

    def _vtk_points_to_numpy(self, vtk_points):
        """Convert vtkPoints to numpy array with safety checks"""
        if not vtk_points:
            return np.zeros((0, 3))
            
        n_points = vtk_points.GetNumberOfPoints()
        if n_points == 0:
            return np.zeros((0, 3))
            
        # More efficient approach using GetData() when available
        if hasattr(vtk_points, 'GetData'):
            points_data = vtk_points.GetData()
            if points_data:
                try:
                    numpy_array = numpy_support.vtk_to_numpy(points_data)
                    return numpy_array.reshape(-1, 3)
                except:
                    pass  # Fall back to manual method if this fails
        
        # Manual conversion as fallback
        points_array = np.zeros((n_points, 3))
        for i in range(n_points):
            points_array[i] = vtk_points.GetPoint(i)
            
        return points_array