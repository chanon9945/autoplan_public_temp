import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca
import logging
import slicer

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
            
            # Add the debug call here
            self.debug_centroid_calculation(self.mask_array, volume)
            
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
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Set fallback values for critical properties
            self.centroid = np.array(insertion_point)
            self.mask_array = np.zeros((1, 1, 1))
            self.volume_array = np.zeros((1, 1, 1))
            self.masked_volume_array = np.zeros((1, 1, 1))
            self.maskedVolume = vtk.vtkImageData()
            self.spinal_canal = np.zeros((1, 1))
            self.pedicle_array = np.zeros((1, 1, 1))
            self.pedicle_mask_array = np.zeros((1, 1, 1))
            self.pedicle = vtk.vtkImageData()
            self.pedicle_mask = vtk.vtkImageData()
            self.pedicle_roi_array = np.zeros((1, 1, 1))
            self.pedicle_roi_mask_array = np.zeros((1, 1, 1))
            self.pedicle_roi = vtk.vtkImageData()
            self.pedicle_roi_mask = vtk.vtkImageData()
            self.pedicle_point_cloud = vtk.vtkPolyData()
            self.pcaVectors = np.eye(3)
            self.pedicle_center_point = np.array(insertion_point)
            self.surface_array = np.zeros((1, 1, 1))
            self.surface = vtk.vtkImageData()
            self.point_cloud = vtk.vtkPolyData()
            raise

    def debug_centroid_calculation(self, mask_array, volume_node):
        """
        Debug function to investigate centroid calculation issues
        """
        # 1. Get dimensions and shape information
        dims = mask_array.shape
        self.logger.info(f"Mask array shape: {dims}")
        
        # 2. Count non-zero voxels in the mask
        non_zero_count = np.count_nonzero(mask_array)
        self.logger.info(f"Non-zero voxels in mask: {non_zero_count}")
        
        # 3. Calculate the centroid in index space
        indices = np.where(mask_array > 0)
        if len(indices[0]) == 0:
            self.logger.warning("No non-zero voxels for centroid calculation")
            return
            
        centroid_i = np.mean(indices[0])
        centroid_j = np.mean(indices[1])
        centroid_k = np.mean(indices[2])
        self.logger.info(f"Centroid in IJK coords: [{centroid_i}, {centroid_j}, {centroid_k}]")
        
        # 4. Check transformations
        if volume_node:
            # Print volume node information
            spacing = volume_node.GetSpacing()
            origin = volume_node.GetOrigin()
            self.logger.info(f"Volume spacing: {spacing}")
            self.logger.info(f"Volume origin: {origin}")
            
            # Get transformation matrices
            ijk_to_ras = vtk.vtkMatrix4x4()
            volume_node.GetIJKToRASMatrix(ijk_to_ras)
            
            # Print matrix elements for debugging
            matrix_elements = [[ijk_to_ras.GetElement(i, j) for j in range(4)] for i in range(4)]
            self.logger.info(f"IJK to RAS matrix: {matrix_elements}")
            
            # Convert IJK to RAS
            ras_coords = [0, 0, 0, 1]
            ijk_coords = [centroid_i, centroid_j, centroid_k, 1]
            ijk_to_ras.MultiplyPoint(ijk_coords, ras_coords)
            self.logger.info(f"Centroid in RAS coords: [{ras_coords[0]}, {ras_coords[1]}, {ras_coords[2]}]")
            
            # Check volume bounds
            bounds = [0, 0, 0, 0, 0, 0]
            volume_node.GetRASBounds(bounds)
            self.logger.info(f"Volume RAS bounds: {bounds}")
            
            # Check if centroid is inside bounds
            inside_x = bounds[0] <= ras_coords[0] <= bounds[1]
            inside_y = bounds[2] <= ras_coords[1] <= bounds[3]
            inside_z = bounds[4] <= ras_coords[2] <= bounds[5]
            self.logger.info(f"Centroid inside bounds: x={inside_x}, y={inside_y}, z={inside_z}")

    def _find_segment_for_level(self, segmentation_node, level):
        """Find the segment ID for a specific vertebra level"""
        if not segmentation_node:
            return None
            
        segmentation = segmentation_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segment_id)
            segment_name = segment.GetName().lower()
            
            # Check if this segment matches the level
            # This needs to be adjusted based on your segment naming convention
            if level.lower() in segment_name or f"{level} vertebra".lower() in segment_name:
                return segment_id
                
        return None

    def _vtk_to_numpy_3d(self, vtk_image):
        """
        Convert vtkImageData or vtkMRMLVolumeNode to 3D numpy array with proper handling
        of both node types and error conditions.
        """
        if vtk_image is None:
            self.logger.warning("Empty VTK image provided")
            return np.zeros((1, 1, 1))
        
        # Determine if this is a MRML node or direct vtkImageData
        image_data = None
        if hasattr(vtk_image, 'GetClassName'):
            class_name = vtk_image.GetClassName()
            
            # For MRML volume nodes, get the underlying image data
            if 'vtkMRMLVolumeNode' in class_name or 'vtkMRMLLabelMapVolumeNode' in class_name:
                image_data = vtk_image.GetImageData()
                if image_data is None:
                    self.logger.warning(f"{class_name} has no image data")
                    return np.zeros((1, 1, 1))
            elif 'vtkImageData' in class_name:
                # It's already a vtkImageData
                image_data = vtk_image
            else:
                self.logger.warning(f"Unsupported input type: {class_name}")
                return np.zeros((1, 1, 1))
        else:
            # If no GetClassName method, assume it's a vtkImageData
            # (less safe but accommodates different VTK Python wrappings)
            image_data = vtk_image
        
        # Now process the vtkImageData
        try:
            dims = image_data.GetDimensions()
            if 0 in dims:
                self.logger.warning(f"VTK image has zero dimension: {dims}")
                return np.zeros((1, 1, 1))
                
            point_data = image_data.GetPointData()
            if not point_data:
                self.logger.warning("VTK image has no point data")
                return np.zeros(dims[::-1])
                
            scalar_data = point_data.GetScalars()
            if not scalar_data:
                self.logger.warning("VTK image has no scalar data")
                return np.zeros(dims[::-1])
                
            # Convert to numpy array
            from vtkmodules.util import numpy_support
            numpy_data = numpy_support.vtk_to_numpy(scalar_data)
            
            # VTK stores data in [z,y,x] order, but we want [x,y,z]
            reshaped_data = numpy_data.reshape(dims[2], dims[1], dims[0])
            return reshaped_data.transpose(2, 1, 0)  # Transpose to [x,y,z]
        except Exception as e:
            self.logger.error(f"Failed to convert VTK data to numpy: {str(e)}")
            if hasattr(image_data, 'GetDimensions'):
                try:
                    dims = image_data.GetDimensions()
                    return np.zeros(dims[::-1])
                except:
                    pass
            return np.zeros((1, 1, 1))

    def _numpy_to_vtk_3d(self, numpy_array, reference_vtk=None):
        """
        Convert 3D numpy array to vtkImageData with proper handling of reference
        objects that may be either MRML nodes or direct vtkImageData.
        """
        if numpy_array is None or numpy_array.size == 0:
            self.logger.warning("Empty numpy array provided")
            return vtk.vtkImageData()
        
        # Create output VTK image
        vtk_image = vtk.vtkImageData()
        
        # Extract image properties from reference if available
        if reference_vtk is not None:
            # Determine if reference is a MRML node or direct vtkImageData
            ref_image_data = None
            ref_spacing = [1.0, 1.0, 1.0]
            ref_origin = [0.0, 0.0, 0.0]
            
            if hasattr(reference_vtk, 'GetClassName'):
                class_name = reference_vtk.GetClassName()
                
                # Handle MRML volume nodes
                if 'vtkMRMLVolumeNode' in class_name or 'vtkMRMLLabelMapVolumeNode' in class_name:
                    ref_image_data = reference_vtk.GetImageData()
                    ref_spacing = reference_vtk.GetSpacing()
                    ref_origin = reference_vtk.GetOrigin()
                
                # Handle direct vtkImageData
                elif 'vtkImageData' in class_name:
                    ref_image_data = reference_vtk
                    ref_spacing = reference_vtk.GetSpacing()
                    ref_origin = reference_vtk.GetOrigin()
            else:
                # Assume it's a vtkImageData if no GetClassName
                ref_image_data = reference_vtk
                if hasattr(reference_vtk, 'GetSpacing'):
                    ref_spacing = reference_vtk.GetSpacing()
                if hasattr(reference_vtk, 'GetOrigin'):
                    ref_origin = reference_vtk.GetOrigin()
            
            # Apply properties to output image
            if ref_image_data and hasattr(ref_image_data, 'GetDimensions'):
                vtk_image.SetDimensions(ref_image_data.GetDimensions())
            else:
                vtk_image.SetDimensions(numpy_array.shape)
                
            vtk_image.SetSpacing(ref_spacing)
            vtk_image.SetOrigin(ref_origin)
            
            # Direction matrix handling is removed to prevent errors
        else:
            # No reference provided, use array shape
            vtk_image.SetDimensions(numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
            vtk_image.SetSpacing(1.0, 1.0, 1.0)
            vtk_image.SetOrigin(0.0, 0.0, 0.0)
        
        try:
            # Transpose numpy array from [x,y,z] to [z,y,x] for VTK
            numpy_array_t = numpy_array.transpose(2, 1, 0).copy()
            flat_array = numpy_array_t.ravel()
            
            # Create VTK array and set as image scalars
            from vtkmodules.util import numpy_support
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
                self.logger.warning("No voxels above threshold found for centroid calculation")
                return None
                
            # Calculate centroid in index space
            centroid_i = np.mean(indices[0])
            centroid_j = np.mean(indices[1])
            centroid_k = np.mean(indices[2])
            
            # Convert to physical coordinates based on available information
            if hasattr(self, 'volume') and self.volume is not None:
                # Determine if this is a MRML node or direct vtkImageData
                if hasattr(self.volume, 'GetClassName'):
                    class_name = self.volume.GetClassName()
                    
                    # Handle MRML volume nodes
                    if 'vtkMRMLVolumeNode' in class_name or 'vtkMRMLLabelMapVolumeNode' in class_name:
                        # Use the transformation matrix
                        ijk_to_ras = vtk.vtkMatrix4x4()
                        self.volume.GetIJKToRASMatrix(ijk_to_ras)
                        
                        # Convert IJK coordinates to RAS
                        ras_coords = [0, 0, 0, 1]  # Homogeneous coordinates
                        ijk_point = [centroid_i, centroid_j, centroid_k, 1]
                        ijk_to_ras.MultiplyPoint(ijk_point, ras_coords)
                        
                        return [ras_coords[0], ras_coords[1], ras_coords[2]]
                    
                    # Handle direct vtkImageData
                    elif 'vtkImageData' in class_name:
                        spacing = self.volume.GetSpacing()
                        origin = self.volume.GetOrigin()
                        
                        # Simple coordinate transformation
                        x = origin[0] + centroid_i * spacing[0]
                        y = origin[1] + centroid_j * spacing[1]
                        z = origin[2] + centroid_k * spacing[2]
                        
                        return [x, y, z]
                
                # As a fallback, try to get spacing and origin directly
                if hasattr(self.volume, 'GetSpacing') and hasattr(self.volume, 'GetOrigin'):
                    spacing = self.volume.GetSpacing()
                    origin = self.volume.GetOrigin()
                    
                    # Simple coordinate transformation
                    x = origin[0] + centroid_i * spacing[0]
                    y = origin[1] + centroid_j * spacing[1]
                    z = origin[2] + centroid_k * spacing[2]
                    
                    return [x, y, z]
            
            # Fallback to index coordinates
            self.logger.warning("Using index coordinates for centroid (no proper transformation available)")
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
    
    def debug_centroid_calculation(self, mask_array, volume_node):
        """
        Debug function to investigate centroid calculation issues
        """
        # 1. Get dimensions and shape information
        dims = mask_array.shape
        self.logger.info(f"Mask array shape: {dims}")
        
        # 2. Count non-zero voxels in the mask
        non_zero_count = np.count_nonzero(mask_array)
        self.logger.info(f"Non-zero voxels in mask: {non_zero_count}")
        
        # 3. Calculate the centroid in index space
        indices = np.where(mask_array > 0)
        if len(indices[0]) == 0:
            self.logger.warning("No non-zero voxels for centroid calculation")
            return
            
        centroid_i = np.mean(indices[0])
        centroid_j = np.mean(indices[1])
        centroid_k = np.mean(indices[2])
        self.logger.info(f"Centroid in IJK coords: [{centroid_i}, {centroid_j}, {centroid_k}]")
        
        # 4. Check the type of volume_node and handle accordingly
        if volume_node is None:
            self.logger.warning("Volume node is None - cannot perform coordinate transformation")
            return
            
        # For MRML nodes
        if hasattr(volume_node, 'GetClassName'):
            class_name = volume_node.GetClassName()
            self.logger.info(f"Volume node class: {class_name}")
            
            # Handle MRML volume nodes
            if 'vtkMRMLVolumeNode' in class_name or 'vtkMRMLLabelMapVolumeNode' in class_name:
                # Get the underlying image data
                image_data = volume_node.GetImageData()
                if image_data is None:
                    self.logger.warning("Volume node has no image data")
                    return
                    
                # Get spacing and origin from the MRML node
                spacing = volume_node.GetSpacing()
                origin = volume_node.GetOrigin()
                self.logger.info(f"Volume spacing: {spacing}")
                self.logger.info(f"Volume origin: {origin}")
                
                # Get dimensions from the image data
                dimensions = image_data.GetDimensions()
                self.logger.info(f"Image dimensions: {dimensions}")
                
                # Get transformation matrices from the MRML node
                ijk_to_ras = vtk.vtkMatrix4x4()
                volume_node.GetIJKToRASMatrix(ijk_to_ras)
                
                # Print matrix elements for debugging
                matrix_elements = [[ijk_to_ras.GetElement(i, j) for j in range(4)] for i in range(4)]
                self.logger.info(f"IJK to RAS matrix: {matrix_elements}")
                
                # Convert IJK to RAS
                ras_coords = [0, 0, 0, 1]
                ijk_coords = [centroid_i, centroid_j, centroid_k, 1]
                ijk_to_ras.MultiplyPoint(ijk_coords, ras_coords)
                self.logger.info(f"Centroid in RAS coords: [{ras_coords[0]}, {ras_coords[1]}, {ras_coords[2]}]")
                
                # Check volume bounds
                bounds = [0, 0, 0, 0, 0, 0]
                volume_node.GetRASBounds(bounds)
                self.logger.info(f"Volume RAS bounds: {bounds}")
                
                # Check if centroid is inside bounds
                inside_x = bounds[0] <= ras_coords[0] <= bounds[1]
                inside_y = bounds[2] <= ras_coords[1] <= bounds[3]
                inside_z = bounds[4] <= ras_coords[2] <= bounds[5]
                self.logger.info(f"Centroid inside bounds: x={inside_x}, y={inside_y}, z={inside_z}")
                
            # Direct vtkImageData
            elif 'vtkImageData' in class_name:
                self.logger.info("Volume node is a vtkImageData")
                
                spacing = volume_node.GetSpacing()
                origin = volume_node.GetOrigin()
                dimensions = volume_node.GetDimensions()
                self.logger.info(f"Image spacing: {spacing}")
                self.logger.info(f"Image origin: {origin}")
                self.logger.info(f"Image dimensions: {dimensions}")
                
                # Calculate approximate RAS coordinates
                approx_x = origin[0] + centroid_i * spacing[0]
                approx_y = origin[1] + centroid_j * spacing[1]
                approx_z = origin[2] + centroid_k * spacing[2]
                self.logger.info(f"Approximate centroid in physical coords: [{approx_x}, {approx_y}, {approx_z}]")
                
                # Get image extent
                extent = volume_node.GetExtent()
                self.logger.info(f"Image extent: {extent}")
            else:
                self.logger.warning(f"Unknown volume node class: {class_name}")
        else:
            self.logger.warning("Volume node has no GetClassName method - cannot determine type")
    
def visualize_critical_points(vertebra):
    """
    Create visual markers for important points used in trajectory planning.
    
    Parameters:
        vertebra: The Vertebra object containing the critical points
    """
    import slicer
    import logging
    import vtk
    
    try:
        # Create a markups fiducial node for visualization if it doesn't exist
        debug_fiducials_name = "TrajectoryDebugPoints"
        debug_fiducials = slicer.mrmlScene.GetFirstNodeByName(debug_fiducials_name)
        
        if not debug_fiducials:
            debug_fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_fiducials_name)
            debug_fiducials.CreateDefaultDisplayNodes()
            
            # Configure display properties
            display_node = debug_fiducials.GetDisplayNode()
            if display_node:
                display_node.SetTextScale(3.0)
                display_node.SetGlyphScale(3.0)
                display_node.SetSelectedColor(1.0, 1.0, 0.0)  # Yellow for selection
        else:
            # Clear existing fiducials
            debug_fiducials.RemoveAllMarkups()
        
        # Add the insertion point (use the one we know is correct)
        insertion_point = vertebra.insertion_point
        idx1 = debug_fiducials.AddFiducial(insertion_point[0], insertion_point[1], insertion_point[2])
        debug_fiducials.SetNthFiducialLabel(idx1, "Insertion Point")
        debug_fiducials.SetNthFiducialSelected(idx1, False)
        debug_fiducials.SetNthFiducialLocked(idx1, True)
        debug_fiducials.SetNthFiducialVisibility(idx1, True)
        
        # Set color for insertion point fiducial
        display_node = debug_fiducials.GetDisplayNode()
        display_node.SetColor(0.0, 1.0, 0.0)  # Green
        
        # Add the main centroid if it exists
        if hasattr(vertebra, 'centroid') and vertebra.centroid is not None:
            centroid = vertebra.centroid
            idx2 = debug_fiducials.AddFiducial(centroid[0], centroid[1], centroid[2])
            debug_fiducials.SetNthFiducialLabel(idx2, "Volume Centroid")
            debug_fiducials.SetNthFiducialSelected(idx2, False)
            debug_fiducials.SetNthFiducialLocked(idx2, True)
            debug_fiducials.SetNthFiducialVisibility(idx2, True)
            
            # We can't set individual point colors in older Slicer versions
            # Just log it so we know which is which
            logging.info(f"Volume Centroid (RED in 3D view): {centroid}")
        
        # Add the pedicle centroid if it exists
        if hasattr(vertebra, 'pedicle_center_point') and vertebra.pedicle_center_point is not None:
            pedicle_center = vertebra.pedicle_center_point
            idx3 = debug_fiducials.AddFiducial(pedicle_center[0], pedicle_center[1], pedicle_center[2])
            debug_fiducials.SetNthFiducialLabel(idx3, "Pedicle Center")
            debug_fiducials.SetNthFiducialSelected(idx3, False)
            debug_fiducials.SetNthFiducialLocked(idx3, True)
            debug_fiducials.SetNthFiducialVisibility(idx3, True)
            
            logging.info(f"Pedicle Center (BLUE in 3D view): {pedicle_center}")
            
        # Log the coordinates for reference
        logging.info(f"Insertion Point (GREEN in 3D view): {insertion_point}")
        
        # Add the predicted trajectory line
        if hasattr(vertebra, 'pcaVectors') and vertebra.pcaVectors is not None:
            # Show the primary direction from PCA
            pca_direction = vertebra.pcaVectors[:, 2]  # Third principal component
            line_length = 30  # mm
            
            # Create a line from pedicle center along PCA direction
            if hasattr(vertebra, 'pedicle_center_point') and vertebra.pedicle_center_point is not None:
                start_point = vertebra.pedicle_center_point
            else:
                start_point = vertebra.insertion_point
                
            end_point = [
                start_point[0] + pca_direction[0] * line_length,
                start_point[1] + pca_direction[1] * line_length,
                start_point[2] + pca_direction[2] * line_length
            ]
            
            # Create a markups line
            pca_line_name = "PCA_Direction"
            pca_line = slicer.mrmlScene.GetFirstNodeByName(pca_line_name)
            if not pca_line:
                pca_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", pca_line_name)
                pca_line.CreateDefaultDisplayNodes()
                
                # Set line properties
                line_display = pca_line.GetDisplayNode()
                if line_display:
                    line_display.SetColor(1.0, 0.5, 0.0)  # Orange
                    line_display.SetLineThickness(3.0)
            else:
                pca_line.RemoveAllMarkups()
                
            # Add the line points
            pca_line.AddControlPoint(vtk.vtkVector3d(start_point))
            pca_line.AddControlPoint(vtk.vtkVector3d(end_point))
            
            logging.info(f"PCA Direction (ORANGE line in 3D view) from {start_point} to {end_point}")
            
        # Adjust view to show the points
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        
        # Show the markups in all viewers
        for view_name in ["Red", "Yellow", "Green"]:
            slice_widget = slicer.app.layoutManager().sliceWidget(view_name)
            if slice_widget:
                slice_logic = slice_widget.sliceLogic()
                if slice_logic:
                    # Center on insertion point
                    try:
                        slice_logic.SetSliceOffset(insertion_point[0 if view_name == "Yellow" else 1 if view_name == "Green" else 2])
                    except:
                        # Alternative method if the above fails
                        slice_node = slice_logic.GetSliceNode()
                        slice_node.JumpSlice(insertion_point[0], insertion_point[1], insertion_point[2])
        
        return debug_fiducials
        
    except Exception as e:
        import traceback
        logging.error(f"Error visualizing critical points: {str(e)}")
        logging.error(traceback.format_exc())
        return None