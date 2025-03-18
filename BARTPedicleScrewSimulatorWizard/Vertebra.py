import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca
import logging
import slicer

class Vertebra:
    """
    Class representing a vertebra for pedicle screw trajectory planning.
    Handles coordinate transforms, segmentation processing, and anatomical analysis.
    """
    
    def __init__(self, segmentation_node, volume_node, insertion_point):
        """
        Initialize a Vertebra object from segmentation and volume nodes.
        
        Parameters:
            segmentation_node: vtkMRMLSegmentationNode or vtkMRMLLabelMapVolumeNode
            volume_node: vtkMRMLScalarVolumeNode containing CT intensities
            insertion_point: [x, y, z] coordinates of surgeon-specified insertion point
        """
        self.logger = logging.getLogger(__name__)
        self.insertion_point = np.array(insertion_point)
        
        try:
            # Create temporary labelmap if segmentation_node is a segmentation
            temp_labelmap_node = None
            if hasattr(segmentation_node, 'GetClassName'):
                class_name = segmentation_node.GetClassName()
                if 'vtkMRMLSegmentationNode' in class_name:
                    self.logger.info("Converting segmentation to labelmap")
                    temp_labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TempLabelmap")
                    
                    # Export visible segments to the labelmap
                    success = slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                        segmentation_node, temp_labelmap_node, volume_node)
                    
                    if not success:
                        self.logger.error("Failed to export segmentation to labelmap")
                        raise RuntimeError("Segmentation export failed")
                    
                    # Use the temporary labelmap for processing
                    self.mask_node = temp_labelmap_node
                else:
                    # It's already a labelmap or other type
                    self.mask_node = segmentation_node
            else:
                # Assume it's a data object (not a MRML node)
                self.mask_node = segmentation_node
            
            # Check volume node type
            if hasattr(volume_node, 'GetClassName'):
                class_name = volume_node.GetClassName()
                self.logger.info(f"Volume node class: {class_name}")
                
                # If it's a MRML volume node, we're good
                if 'vtkMRMLVolumeNode' in class_name:
                    self.volume_node = volume_node
                else:
                    # It's probably a vtkImageData, not a volume node
                    self.logger.warning(f"Expected vtkMRMLVolumeNode but got {class_name}")
                    # We'll still use it, but some operations may fail
                    self.volume_node = volume_node
            else:
                # Assume it's a vtkImageData or similar
                self.logger.warning("Volume node doesn't have GetClassName method")
                self.volume_node = volume_node
            
            # Convert nodes to numpy arrays for processing
            self.mask_array = self._volume_to_numpy_array(self.mask_node)
            self.volume_array = self._volume_to_numpy_array(self.volume_node)
            
            # Create masked volume (volume × mask)
            self.masked_volume_array = self.volume_array * (self.mask_array > 0).astype(float)
            
            # Make masked volume available to other components
            self.maskedVolume = self._numpy_to_volume(self.masked_volume_array, self.volume_node)
            
            # Calculate centroid of the vertebra
            self.centroid = self._calculate_centroid(self.mask_array)
            self.logger.info(f"Vertebra centroid: {self.centroid}")
            
            # Find spinal canal
            canal_min_y, canal_max_y = self._detect_spinal_canal()
            
            # Cut out pedicle region
            self.pedicle_array = self._cut_pedicle(
                self.masked_volume_array.copy(), 
                canal_max_y, 
                canal_min_y, 
                buffer_front=15,
                buffer_end=1
            )
            
            # Separate relevant side (left/right based on insertion point)
            self.pedicle_roi_array = self._cut_pedicle_side(
                self.pedicle_array.copy(),
                self.insertion_point[0],
                self.centroid[0]
            )
            
            # Create point cloud for the pedicle
            self.pedicle_point_cloud = self._array_to_point_cloud(
                self.pedicle_roi_array, 
                self.volume_node,
                threshold=0
            )
            
            # Perform PCA on pedicle points
            if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
                points_array = self._points_to_numpy(self.pedicle_point_cloud.GetPoints())
                if points_array.shape[0] >= 3:
                    # Run PCA
                    self.coeff, self.latent, self.score = apply_pca(points_array)
                    self.pedicle_center_point = np.mean(points_array, axis=0)
                    
                    # Scale eigenvectors by eigenvalues for visualization
                    scaling_factor = np.sqrt(self.latent) * 2
                    self.pcaVectors = self.coeff * scaling_factor[:, np.newaxis]
                    self.logger.info(f"PCA principal axis: {self.pcaVectors[:, 2]}")
                else:
                    self.logger.warning(f"Not enough points for PCA: {points_array.shape}")
                    self._set_default_values()
            else:
                self.logger.warning("No pedicle point cloud available for PCA")
                self._set_default_values()
            
            # Extract surface points for collision detection
            self.surface_array = self._extract_surface_voxels(self.mask_array)
            self.point_cloud = self._array_to_point_cloud(
                self.surface_array,
                self.volume_node,
                threshold=0
            )
            
            # Clean up temporary node
            if temp_labelmap_node:
                slicer.mrmlScene.RemoveNode(temp_labelmap_node)
                
        except Exception as e:
            self.logger.error(f"Error initializing Vertebra: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Set default values on error
            self._set_default_values()
            
            # Re-raise for caller
            raise
    
    def _set_default_values(self):
        """Set default values when processing fails"""
        if not hasattr(self, 'centroid') or self.centroid is None:
            self.centroid = self.insertion_point
            
        if not hasattr(self, 'pcaVectors') or self.pcaVectors is None:
            self.pcaVectors = np.eye(3)
            
        if not hasattr(self, 'pedicle_center_point') or self.pedicle_center_point is None:
            self.pedicle_center_point = self.insertion_point

    def _volume_to_numpy_array(self, volume_node):
        """
        Convert a volume node to a numpy array, handling coordinate transforms.
        
        Parameters:
            volume_node: vtkMRMLVolumeNode
            
        Returns:
            numpy.ndarray: 3D array of voxel values
        """
        if not volume_node:
            self.logger.warning("Empty volume node")
            return np.zeros((1, 1, 1))
        
        # Get image data
        image_data = None
        if hasattr(volume_node, 'GetImageData'):
            image_data = volume_node.GetImageData()
        else:
            self.logger.warning("Volume node has no GetImageData method")
            return np.zeros((1, 1, 1))
        
        if not image_data:
            self.logger.warning("No image data in volume node")
            return np.zeros((1, 1, 1))
        
        # Get dimensions
        dims = image_data.GetDimensions()
        if 0 in dims:
            self.logger.warning(f"Volume has zero dimension: {dims}")
            return np.zeros((1, 1, 1))
        
        # Get scalars
        scalars = image_data.GetPointData().GetScalars()
        if not scalars:
            self.logger.warning("No scalar data in volume")
            return np.zeros(dims[::-1])
        
        # Convert to numpy array
        numpy_array = numpy_support.vtk_to_numpy(scalars)
        
        # Reshape to 3D array (VTK's order is [z,y,x], we want [x,y,z])
        reshaped_array = numpy_array.reshape(dims[2], dims[1], dims[0])
        
        # Transpose to match our expected order
        return reshaped_array.transpose(2, 1, 0)
    
    def _numpy_to_volume(self, numpy_array, reference_node):
        """
        Convert a numpy array to a vtkMRMLScalarVolumeNode.
        
        Parameters:
            numpy_array: 3D numpy array
            reference_node: Node to copy geometry from
            
        Returns:
            vtkMRMLScalarVolumeNode: Volume node containing the data
        """
        if numpy_array is None or numpy_array.size == 0:
            self.logger.warning("Empty numpy array provided")
            return None
        
        # Create a new VTK image data
        img_vtk = vtk.vtkImageData()
        
        # Set dimensions and other properties from reference
        if reference_node and hasattr(reference_node, 'GetImageData'):
            ref_image = reference_node.GetImageData()
            if ref_image:
                img_vtk.SetDimensions(ref_image.GetDimensions())
                img_vtk.SetSpacing(ref_image.GetSpacing())
                img_vtk.SetOrigin(ref_image.GetOrigin())
        else:
            img_vtk.SetDimensions(*numpy_array.shape)
            img_vtk.SetSpacing(1.0, 1.0, 1.0)
            img_vtk.SetOrigin(0.0, 0.0, 0.0)
        
        # Transpose array to VTK's ordering [z,y,x]
        vtk_array = numpy_array.transpose(2, 1, 0).copy()
        
        # Convert to VTK array
        flat_array = vtk_array.ravel()
        vtk_data_array = numpy_support.numpy_to_vtk(
            flat_array,
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        
        # Set the array as scalars in the image data
        img_vtk.GetPointData().SetScalars(vtk_data_array)
        
        # Create a volume node
        if hasattr(slicer, 'mrmlScene'):
            # Use Slicer API if available
            volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ProcessedVolume")
            volume_node.SetAndObserveImageData(img_vtk)
            
            # Copy geometry from reference
            if reference_node and hasattr(reference_node, 'GetIJKToRASMatrix'):
                ijk_to_ras = vtk.vtkMatrix4x4()
                reference_node.GetIJKToRASMatrix(ijk_to_ras)
                volume_node.SetIJKToRASMatrix(ijk_to_ras)
            
            return volume_node
        else:
            # Just return the image data if not in Slicer
            return img_vtk
    
    def _calculate_centroid(self, mask_array, threshold=0):
        """
        Calculate the centroid of non-zero voxels in a binary mask.
        
        Parameters:
            mask_array: 3D numpy array
            threshold: Minimum value to consider as part of the object
            
        Returns:
            numpy.ndarray: [x, y, z] coordinates of centroid
        """
        # Find indices of non-zero voxels
        indices = np.where(mask_array > threshold)
        
        if len(indices[0]) == 0:
            self.logger.warning("No voxels above threshold for centroid calculation")
            return self.insertion_point
        
        # Calculate mean position in IJK coordinates
        centroid_ijk = np.array([
            np.mean(indices[0]),
            np.mean(indices[1]),
            np.mean(indices[2])
        ])
        
        # Convert from IJK to RAS if volume_node is available
        if hasattr(self, 'volume_node') and self.volume_node:
            # Check if we have a MRML volume node or a vtkImageData
            if hasattr(self.volume_node, 'GetIJKToRASMatrix'):
                # It's a MRML volume node
                ijk_to_ras = vtk.vtkMatrix4x4()
                self.volume_node.GetIJKToRASMatrix(ijk_to_ras)
                
                # Convert to RAS coordinates
                ijk_coord = np.append(centroid_ijk, 1.0)  # Homogeneous coordinates
                ras_coord = [0, 0, 0, 1]
                ijk_to_ras.MultiplyPoint(ijk_coord, ras_coord)
                
                return np.array(ras_coord[0:3])
            else:
                # It's a vtkImageData or something else
                self.logger.info("Volume node doesn't have GetIJKToRASMatrix - using spacing and origin")
                
                # Try to get spacing and origin directly
                if hasattr(self.volume_node, 'GetSpacing') and hasattr(self.volume_node, 'GetOrigin'):
                    spacing = self.volume_node.GetSpacing()
                    origin = self.volume_node.GetOrigin()
                    
                    # Apply simple transform: RAS = IJK * spacing + origin
                    ras_x = centroid_ijk[0] * spacing[0] + origin[0]
                    ras_y = centroid_ijk[1] * spacing[1] + origin[1]
                    ras_z = centroid_ijk[2] * spacing[2] + origin[2]
                    
                    return np.array([ras_x, ras_y, ras_z])
                else:
                    self.logger.warning("Cannot get spacing/origin - returning IJK coordinates")
        
        # Return IJK coordinates if we can't convert to RAS
        return centroid_ijk
    
    def _detect_spinal_canal(self):
        """
        Detect the anterior and posterior boundaries of the spinal canal.
        
        Returns:
            tuple: (canal_min_y, canal_max_y) in image coordinates
        """
        # If no centroid, return default values
        if not hasattr(self, 'centroid') or self.centroid is None:
            canal_min_y = 0
            canal_max_y = self.mask_array.shape[1] // 2
            return canal_min_y, canal_max_y
        
        # Convert centroid to IJK if it's in RAS
        centroid_ijk = self.centroid
        if hasattr(self, 'volume_node') and self.volume_node:
            # Convert from RAS to IJK
            ras_to_ijk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(ras_to_ijk)
            
            ras_coord = np.append(self.centroid, 1.0)
            ijk_coord = [0, 0, 0, 1]
            ras_to_ijk.MultiplyPoint(ras_coord, ijk_coord)
            
            centroid_ijk = np.array(ijk_coord[0:3])
        
        # Get mid-axial slice at centroid
        sliceZ = int(np.round(centroid_ijk[2]))
        sliceZ = max(0, min(sliceZ, self.mask_array.shape[2] - 1))
        
        # Extract slice
        slice_mask = self.mask_array[:, :, sliceZ].copy()
        
        # Find posterior boundary by stepping backward from centroid
        posterior_y = int(np.round(centroid_ijk[1]))
        while posterior_y > 0:
            if slice_mask[int(np.round(centroid_ijk[0])), posterior_y] == 0:
                break
            posterior_y -= 1
        
        # Find anterior boundary by stepping forward from posterior
        anterior_y = posterior_y
        while anterior_y < slice_mask.shape[1] - 1:
            if slice_mask[int(np.round(centroid_ijk[0])), anterior_y] > 0:
                break
            anterior_y += 1
        
        # Add margin to make sure we fully capture the canal
        canal_min_y = max(0, posterior_y - 5)
        canal_max_y = min(slice_mask.shape[1] - 1, anterior_y + 5)
        
        return canal_min_y, canal_max_y
    
    def _cut_pedicle(self, volume_array, y_max, y_min, buffer_front=15, buffer_end=1):
        """
        Cut out the pedicle region from the volume.
        
        Parameters:
            volume_array: 3D numpy array
            y_max: Maximum Y coordinate of spinal canal
            y_min: Minimum Y coordinate of spinal canal
            buffer_front: Buffer before spinal canal
            buffer_end: Buffer after spinal canal
            
        Returns:
            numpy.ndarray: Cropped array containing only the pedicle region
        """
        # Create a copy to avoid modifying the original
        result = volume_array.copy()
        
        # Calculate boundaries
        y_min_idx = max(0, min(int(y_min + buffer_front + 1), result.shape[1] - 1))
        y_max_idx = max(0, min(int(y_max - buffer_end), result.shape[1] - 1))
        
        # Zero out regions outside pedicle
        if y_min_idx > 0:
            result[:, 0:y_min_idx, :] = 0
        if y_max_idx < result.shape[1] - 1:
            result[:, y_max_idx:, :] = 0
        
        return result
    
    def _cut_pedicle_side(self, volume_array, insertion_x, centroid_x):
        """
        Cut the volume to keep only the relevant side (left or right).
        
        Parameters:
            volume_array: 3D numpy array
            insertion_x: X coordinate of insertion point
            centroid_x: X coordinate of vertebra centroid
            
        Returns:
            numpy.ndarray: Cropped array containing only one side of the pedicle
        """
        # Create a copy to avoid modifying the original
        result = volume_array.copy()
        
        # Convert coordinates to image space if needed
        insertion_x_ijk = insertion_x
        centroid_x_ijk = centroid_x
        
        if hasattr(self, 'volume_node') and self.volume_node:
            # Check if we have a MRML volume node with transformation matrix
            if hasattr(self.volume_node, 'GetRASToIJKMatrix'):
                # Convert from RAS to IJK
                ras_to_ijk = vtk.vtkMatrix4x4()
                self.volume_node.GetRASToIJKMatrix(ras_to_ijk)
                
                # Convert insertion point
                ins_ras = np.append(np.array([insertion_x, 0, 0]), 1.0)
                ins_ijk = [0, 0, 0, 1]
                ras_to_ijk.MultiplyPoint(ins_ras, ins_ijk)
                insertion_x_ijk = ins_ijk[0]
                
                # Convert centroid
                cent_ras = np.append(np.array([centroid_x, 0, 0]), 1.0)
                cent_ijk = [0, 0, 0, 1]
                ras_to_ijk.MultiplyPoint(cent_ras, cent_ijk)
                centroid_x_ijk = cent_ijk[0]
            else:
                # Try to use spacing and origin
                if hasattr(self.volume_node, 'GetSpacing') and hasattr(self.volume_node, 'GetOrigin'):
                    spacing = self.volume_node.GetSpacing()
                    origin = self.volume_node.GetOrigin()
                    
                    # Convert RAS to IJK: IJK = (RAS - origin) / spacing
                    insertion_x_ijk = (insertion_x - origin[0]) / spacing[0]
                    centroid_x_ijk = (centroid_x - origin[0]) / spacing[0]
        
        # Round to integer for indexing
        centroid_idx = int(np.round(centroid_x_ijk))
        centroid_idx = max(0, min(centroid_idx, result.shape[0] - 1))
        
        # Cut based on which side the insertion point is on
        if insertion_x_ijk > centroid_x_ijk:
            # Keep right side
            if centroid_idx > 0:
                result[0:centroid_idx, :, :] = 0
        else:
            # Keep left side
            if centroid_idx < result.shape[0]:
                result[centroid_idx:, :, :] = 0
        
        return result
    
    def _extract_surface_voxels(self, volume_array):
        """
        Extract the surface voxels of a binary volume.
        
        Parameters:
            volume_array: 3D numpy array
            
        Returns:
            numpy.ndarray: Binary mask of surface voxels
        """
        # Create binary mask
        binary_mask = (volume_array > 0).astype(np.uint8)
        
        # Erode the mask
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(binary_mask)
        
        # Surface is the difference between original and eroded
        surface = np.logical_and(binary_mask, np.logical_not(eroded))
        
        return surface.astype(np.uint8)
    
    def _array_to_point_cloud(self, array, reference_node=None, threshold=0):
        """
        Convert a binary array to a VTK point cloud.
        
        Parameters:
            array: 3D numpy array
            reference_node: Volume node for coordinate transform
            threshold: Minimum value to include
            
        Returns:
            vtkPolyData: Point cloud of non-zero voxels
        """
        # Find indices of voxels above threshold
        indices = np.where(array > threshold)
        
        if len(indices[0]) == 0:
            # Return empty point cloud
            self.logger.warning("No points above threshold")
            points = vtk.vtkPoints()
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            return polydata
        
        # Create points
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(indices[0]))
        
        # Transform indices to physical coordinates
        if reference_node:
            if hasattr(reference_node, 'GetIJKToRASMatrix'):
                # MRML Volume node with transformation matrix
                ijk_to_ras = vtk.vtkMatrix4x4()
                reference_node.GetIJKToRASMatrix(ijk_to_ras)
                
                # Convert each point
                for i in range(len(indices[0])):
                    # Create homogeneous coordinates
                    ijk = [indices[0][i], indices[1][i], indices[2][i], 1.0]
                    ras = [0, 0, 0, 1]
                    ijk_to_ras.MultiplyPoint(ijk, ras)
                    points.SetPoint(i, ras[0], ras[1], ras[2])
            elif hasattr(reference_node, 'GetSpacing') and hasattr(reference_node, 'GetOrigin'):
                # vtkImageData - use spacing and origin
                spacing = reference_node.GetSpacing()
                origin = reference_node.GetOrigin()
                
                # Convert each point: RAS = IJK * spacing + origin
                for i in range(len(indices[0])):
                    x = indices[0][i] * spacing[0] + origin[0]
                    y = indices[1][i] * spacing[1] + origin[1]
                    z = indices[2][i] * spacing[2] + origin[2]
                    points.SetPoint(i, x, y, z)
            else:
                # Use indices directly as coordinates
                for i in range(len(indices[0])):
                    points.SetPoint(i, indices[0][i], indices[1][i], indices[2][i])
        else:
            # Use indices directly as coordinates
            for i in range(len(indices[0])):
                points.SetPoint(i, indices[0][i], indices[1][i], indices[2][i])
        
        # Create polydata with vertex cells
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add vertices (improves rendering)
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        
        polydata.SetVerts(vertices)
        
        return polydata
    
    def _points_to_numpy(self, vtk_points):
        """
        Convert vtkPoints to numpy array.
        
        Parameters:
            vtk_points: vtkPoints object
            
        Returns:
            numpy.ndarray: Array of point coordinates (n_points × 3)
        """
        if not vtk_points:
            return np.zeros((0, 3))
        
        n_points = vtk_points.GetNumberOfPoints()
        if n_points == 0:
            return np.zeros((0, 3))
        
        # Try direct conversion if possible
        try:
            from vtkmodules.util import numpy_support
            if hasattr(vtk_points, 'GetData'):
                points_data = vtk_points.GetData()
                if points_data:
                    numpy_array = numpy_support.vtk_to_numpy(points_data)
                    return numpy_array.reshape(-1, 3)
        except:
            pass
        
        # Manual conversion as fallback
        result = np.zeros((n_points, 3))
        for i in range(n_points):
            result[i] = vtk_points.GetPoint(i)
        
        return result


def visualize_critical_points(vertebra):
    """
    Create visual markers for important points used in trajectory planning.
    
    Parameters:
        vertebra: The Vertebra object containing the critical points
    """
    if not 'slicer' in globals():
        return None  # Not running in Slicer
        
    try:
        # Create a markups fiducial node for visualization
        debug_fiducials_name = "TrajectoryDebugPoints"
        debug_fiducials = slicer.mrmlScene.GetFirstNodeByName(debug_fiducials_name)
        
        if not debug_fiducials:
            debug_fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", debug_fiducials_name)
            debug_fiducials.CreateDefaultDisplayNodes()
            
            # Configure display properties
            display_node = debug_fiducials.GetDisplayNode()
            if display_node:
                display_node.SetTextScale(2.0)
                display_node.SetGlyphScale(2.0)
                display_node.SetSelectedColor(1.0, 1.0, 0.0)  # Yellow for selection
        else:
            # Clear existing fiducials
            debug_fiducials.RemoveAllMarkups()
        
        # Add the insertion point
        insertion_point = vertebra.insertion_point
        idx1 = debug_fiducials.AddFiducial(insertion_point[0], insertion_point[1], insertion_point[2])
        debug_fiducials.SetNthFiducialLabel(idx1, "Insertion Point")
        debug_fiducials.SetNthFiducialSelected(idx1, False)
        debug_fiducials.SetNthFiducialLocked(idx1, True)
        
        # Add the centroid
        if hasattr(vertebra, 'centroid') and vertebra.centroid is not None:
            centroid = vertebra.centroid
            idx2 = debug_fiducials.AddFiducial(centroid[0], centroid[1], centroid[2])
            debug_fiducials.SetNthFiducialLabel(idx2, "Vertebra Centroid")
            debug_fiducials.SetNthFiducialSelected(idx2, False)
            debug_fiducials.SetNthFiducialLocked(idx2, True)
        
        # Add the pedicle center
        if hasattr(vertebra, 'pedicle_center_point') and vertebra.pedicle_center_point is not None:
            pedicle_center = vertebra.pedicle_center_point
            idx3 = debug_fiducials.AddFiducial(pedicle_center[0], pedicle_center[1], pedicle_center[2])
            debug_fiducials.SetNthFiducialLabel(idx3, "Pedicle Center")
            debug_fiducials.SetNthFiducialSelected(idx3, False)
            debug_fiducials.SetNthFiducialLocked(idx3, True)
        
        # Add the PCA axes if available
        if hasattr(vertebra, 'pcaVectors') and vertebra.pcaVectors is not None and hasattr(vertebra, 'pedicle_center_point'):
            center = vertebra.pedicle_center_point
            
            # Create a separate model node for each PCA axis
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB for each principal axis
            labels = ["1st PCA Axis", "2nd PCA Axis", "3rd PCA Axis"]
            
            for i in range(3):
                vector = vertebra.pcaVectors[:, i]
                length = np.linalg.norm(vector)
                if length < 1e-6:
                    continue
                    
                # Scale to reasonable length for visualization
                display_length = 20.0  # mm
                vector = vector / length * display_length
                
                # Create line endpoints
                start_point = center - vector/2
                end_point = center + vector/2
                
                # Create line node
                line_node_name = f"PCA_Axis_{i+1}"
                line_node = slicer.mrmlScene.GetFirstNodeByName(line_node_name)
                if not line_node:
                    line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", line_node_name)
                    line_node.CreateDefaultDisplayNodes()
                else:
                    line_node.RemoveAllMarkups()
                
                # Set line endpoints
                line_node.AddControlPoint(vtk.vtkVector3d(start_point))
                line_node.AddControlPoint(vtk.vtkVector3d(end_point))
                
                # Set display properties
                display_node = line_node.GetDisplayNode()
                if display_node:
                    display_node.SetColor(colors[i])
                    display_node.SetLineThickness(3.0)
                    display_node.SetTextScale(0)  # Hide text
        
        return debug_fiducials
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error visualizing critical points: {str(e)}")
        import traceback
        logging.getLogger(__name__).error(traceback.format_exc())
        return None