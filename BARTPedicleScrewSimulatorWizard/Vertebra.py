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
            if self.centroid is None:
                self.logger.warning("Centroid calculation failed - using fallback")
                # Use fallback - find center of bounding box
                bounds = [0, 0, 0, 0, 0, 0]
                if self.volume:
                    self.volume.GetRASBounds(bounds)
                    self.centroid = [
                        (bounds[0] + bounds[1]) / 2,
                        (bounds[2] + bounds[3]) / 2,
                        (bounds[4] + bounds[5]) / 2
                    ]
                else:
                    self.centroid = np.array(insertion_point)
            
            # Log the centroid
            self.logger.info(f"Volume centroid: {self.centroid}")
            
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
            
            # Calculate pedicle center point explicitly
            if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
                points = self.pedicle_point_cloud.GetPoints()
                num_points = points.GetNumberOfPoints()
                
                # Calculate centroid of pedicle points
                pedicle_center_sum = np.zeros(3)
                for i in range(num_points):
                    point = points.GetPoint(i)
                    pedicle_center_sum += np.array(point)
                    
                if num_points > 0:
                    self.pedicle_center_point = pedicle_center_sum / num_points
                    self.logger.info(f"Calculated pedicle center point: {self.pedicle_center_point}")
                else:
                    self.pedicle_center_point = np.array(self.insertion_point)
                    self.logger.warning("No pedicle points found, using insertion point as fallback")
            else:
                self.pedicle_center_point = np.array(self.insertion_point)
                self.logger.warning("No pedicle point cloud, using insertion point as fallback")
            
            # Run PCA on pedicle point cloud
            if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
                # Convert points to numpy array
                points_array = []
                for i in range(self.pedicle_point_cloud.GetNumberOfPoints()):
                    points_array.append(self.pedicle_point_cloud.GetPoint(i))
                    
                points_array = np.array(points_array)
                
                if points_array.shape[0] >= 3:  # Need at least 3 points for PCA
                    from .PCAUtils import apply_pca
                    self.pcaCoeff, self.pcaLatent, self.pcaScore = apply_pca(points_array)
                    
                    # Scale eigenvectors by eigenvalues for visualization
                    self.pcaVectors = self.pcaCoeff * np.sqrt(self.pcaLatent)[:, np.newaxis]
                    
                    self.logger.info(f"PCA vectors calculated: {self.pcaVectors}")
                    self.logger.info(f"PCA latent values: {self.pcaLatent}")
                else:
                    self.logger.warning("Not enough points for PCA")
                    self.pcaVectors = np.eye(3)  # Identity as fallback
            else:
                self.logger.warning("No pedicle point cloud for PCA")
                self.pcaVectors = np.eye(3)  # Identity as fallback
                
            # Extract surface voxels
            self.surface_array = self._extract_surface_voxels(self.mask_array)
            self.surface = self._numpy_to_vtk_3d(self.surface_array, self.mask)
            self.point_cloud = self._masked_volume_to_point_cloud(self.surface)
            
        except Exception as e:
            import traceback
            self.logger.error(f"Vertebra initialization error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

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