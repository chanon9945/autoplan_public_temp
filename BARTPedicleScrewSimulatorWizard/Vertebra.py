import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca

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
        self.insertion_point = insertion_point
        
        # Multiply volume by mask to get masked volume
        self.maskedVolume = self._multiply_volumes(volume, mask)
        
        # Calculate centroid of masked volume
        self.centroid = self._get_masked_centroid(self.maskedVolume)
        if self.centroid is None:
            raise ValueError("Failed to compute centroid - masked volume may be empty")
        
        # Convert VTK volumes to NumPy arrays for processing
        self.mask_array = self._vtk_to_numpy_3d(mask)
        self.volume_array = self._vtk_to_numpy_3d(volume)
        self.masked_volume_array = self._vtk_to_numpy_3d(self.maskedVolume)
        
        # Process spinal canal
        centroid_slice_z = int(self.centroid[2])
        mask_slice = self.mask_array[:, :, centroid_slice_z]
        
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
        canal_max_y = np.max(stack_y)
        canal_min_y = np.min(stack_y)
        
        # Find canal center
        canal_center, canal_center_x, canal_center_y = self._center(self.spinal_canal)
        
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
        
        if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
            # Convert VTK point cloud to numpy array for PCA
            points_array = self._vtk_points_to_numpy(self.pedicle_point_cloud.GetPoints())
            
            if points_array.shape[0] >= 3:  # Need at least 3 points for meaningful PCA
                coeff, latent, score = apply_pca(points_array)
                pedicle_center_point = np.mean(points_array, axis=0)
                scaling_factor = np.sqrt(latent) * 2
                self.pca_vectors = coeff * scaling_factor
            else:
                # Default values if not enough points
                self.pca_vectors = np.eye(3)
        else:
            # Default values if point cloud creation failed
            self.pca_vectors = np.eye(3)
            
        # Extract surface voxels
        self.surface_array = self._extract_surface_voxels(self.mask_array)
        self.surface = self._numpy_to_vtk_3d(self.surface_array, self.mask)
        self.point_cloud = self._masked_volume_to_point_cloud(self.surface)

    def _multiply_volumes(self, vol1, vol2):
        """Multiply two vtkImageData volumes using vtkImageMathematics"""
        math_filter = vtk.vtkImageMathematics()
        math_filter.SetOperationToMultiply()
        math_filter.SetInputData(0, vol1)
        math_filter.SetInputData(1, vol2)
        math_filter.Update()
        return math_filter.GetOutput()

    def _vtk_to_numpy_3d(self, vtk_image):
        """Convert vtkImageData to 3D numpy array"""
        if not vtk_image:
            return np.zeros((1, 1, 1))
            
        dims = vtk_image.GetDimensions()
        scalar_data = vtk_image.GetPointData().GetScalars()
        
        if not scalar_data:
            return np.zeros(dims[::-1])
            
        numpy_data = numpy_support.vtk_to_numpy(scalar_data)
        
        # VTK stores data in format (z, y, x)
        return numpy_data.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)

    def _numpy_to_vtk_3d(self, numpy_array, reference_vtk=None):
        """Convert 3D numpy array to vtkImageData"""
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
            vtk_image.SetDirectionMatrix(reference_vtk.GetDirectionMatrix())
            
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

    def _get_masked_centroid(self, masked_volume, threshold=0):
        """
        Compute the centroid of nonzero voxels in a masked volume.
        Returns [x, y, z] coordinates in physical space.
        """
        try:
            # Get the dimensions, spacing, and origin
            dims = masked_volume.GetDimensions()
            spacing = masked_volume.GetSpacing()
            origin = masked_volume.GetOrigin()
            
            # Get the voxel data
            scalar_array = masked_volume.GetPointData().GetScalars()
            if not scalar_array:
                return None
                
            numpy_array = numpy_support.vtk_to_numpy(scalar_array)
            numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)
            
            # Find indices where the value is above threshold
            indices = np.where(numpy_array > threshold)
            
            if len(indices[0]) == 0:
                return None
                
            # Calculate centroid in index space
            centroid_i = np.mean(indices[0])
            centroid_j = np.mean(indices[1])
            centroid_k = np.mean(indices[2])
            
            # Convert to physical coordinates
            centroid_x = origin[0] + spacing[0] * centroid_i
            centroid_y = origin[1] + spacing[1] * centroid_j
            centroid_z = origin[2] + spacing[2] * centroid_k
            
            return [centroid_x, centroid_y, centroid_z]
            
        except Exception as e:
            print(f"Error in get_masked_centroid: {e}")
            return None

    def _centroid_shift(self, coord_x, coord_y, coord_z, mask_array):
        """
        Find a point inside the spinal canal by shifting posteriorly.
        Python implementation of centroidShift.m
        """
        # Base case: we're at the edge or found an empty voxel
        if coord_y <= 0:
            return coord_x, coord_y
            
        if mask_array[coord_x, coord_y, coord_z] == 0:
            return coord_x, coord_y
        else:
            # Recursively move backwards along Y
            return self._centroid_shift(coord_x, coord_y - 1, coord_z, mask_array)

    def _floodfill(self, image, i, j):
        """
        Floodfill to find connected region.
        Python implementation of floodfill.m
        """
        image_size = image.shape
        shape = np.zeros(image_size, dtype=image.dtype)
        stack_x = [i]
        stack_y = [j]
        
        filled_shape, stack_x, stack_y = self._dfs(image, image_size, i, j, shape, stack_x, stack_y)
        return filled_shape, np.array(stack_x), np.array(stack_y)

    def _dfs(self, image, image_size, i, j, shape, stack_x, stack_y):
        """
        Depth-first search for floodfill.
        Python implementation of dfs.m
        """
        # Check boundary conditions and if already visited
        if i < 0 or j < 0 or i >= image_size[0] or j >= image_size[1]:
            return shape, stack_x, stack_y
            
        if shape[i, j] != 0 or image[i, j] != 0:
            return shape, stack_x, stack_y
            
        # Mark this pixel
        shape[i, j] = 1
        stack_x.append(i)
        stack_y.append(j)
        
        # Visit neighbors
        shape, stack_x, stack_y = self._dfs(image, image_size, i+1, j, shape, stack_x, stack_y)
        shape, stack_x, stack_y = self._dfs(image, image_size, i-1, j, shape, stack_x, stack_y)
        shape, stack_x, stack_y = self._dfs(image, image_size, i, j+1, shape, stack_x, stack_y)
        shape, stack_x, stack_y = self._dfs(image, image_size, i, j-1, shape, stack_x, stack_y)
        
        return shape, stack_x, stack_y

    def _center(self, shape):
        """
        Compute the center of a 2D binary image.
        Python implementation of center.m
        """
        non_zero_mask = shape > 0
        if not np.any(non_zero_mask):
            return (0, 0), 0, 0
            
        x_indices, y_indices = np.meshgrid(range(shape.shape[0]), range(shape.shape[1]), indexing='ij')
        
        center_x = int(np.round(np.mean(x_indices[non_zero_mask])))
        center_y = int(np.round(np.mean(y_indices[non_zero_mask])))
        
        return (center_x, center_y), center_x, center_y

    def _cut_pedicle(self, vol_in, y_max, y_min, buffer_front=15, buffer_end=1):
        """
        Cut the pedicle using the spinal canal boundaries.
        Python implementation of cutPedicle.m
        """
        vol_in[:, 0:y_min+buffer_front+1, :] = 0
        vol_in[:, y_max-buffer_end:, :] = 0
        return vol_in

    def _pedicle_roi_cut(self, vol, roi_point_x, centroid_x):
        """
        Cut the pedicle ROI based on the insertion point.
        Python implementation of pedicleROICut.m
        """
        if roi_point_x > centroid_x:
            vol[:int(centroid_x), :, :] = 0
        else:
            vol[int(centroid_x):, :, :] = 0
        return vol

    def _extract_surface_voxels(self, volume):
        """
        Extract surface voxels from a binary volume.
        Python implementation of extractSurfaceVoxels.m
        """
        from scipy.ndimage import binary_erosion
        
        # Perform binary erosion
        eroded = binary_erosion(volume)
        
        # Surface voxels are in original but not in eroded
        surface_voxels = np.logical_and(volume, np.logical_not(eroded))
        return surface_voxels

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
            for i in range(len(indices[0])):
                x = origin[0] + indices[0][i] * spacing[0]
                y = origin[1] + indices[1][i] * spacing[1]
                z = origin[2] + indices[2][i] * spacing[2]
                points.InsertNextPoint(x, y, z)
            
            # Create polydata
            point_cloud = vtk.vtkPolyData()
            point_cloud.SetPoints(points)
            
            return point_cloud
            
        except Exception as e:
            print(f"Error in masked_volume_to_point_cloud: {e}")
            # Return an empty point cloud
            points = vtk.vtkPoints()
            point_cloud = vtk.vtkPolyData()
            point_cloud.SetPoints(points)
            return point_cloud

    def _vtk_points_to_numpy(self, vtk_points):
        """Convert vtkPoints to numpy array"""
        n_points = vtk_points.GetNumberOfPoints()
        points_array = np.zeros((n_points, 3))
        
        for i in range(n_points):
            point = vtk_points.GetPoint(i)
            points_array[i] = point
            
        return points_array