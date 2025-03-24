import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca, select_best_alignment_vector
import logging
import slicer
import traceback

class Vertebra:
    """
    Class representing a vertebra for pedicle screw trajectory planning.
    Handles coordinate transforms, segmentation processing, and anatomical analysis.
    """
    
    def __init__(self, segmentation_node, volume_node, insertion_point, target_level=None):
        """
        Initialize a Vertebra object from segmentation and volume nodes.
        
        Parameters:
            segmentation_node: vtkMRMLSegmentationNode or vtkMRMLLabelMapVolumeNode
            volume_node: vtkMRMLScalarVolumeNode containing CT intensities
            insertion_point: [x, y, z] coordinates of surgeon-specified insertion point
            target_level: Target vertebra level (e.g., 'L4', 'L5') - extracted from fiducial if not provided
        """
        self.logger = logging.getLogger(__name__)
        self.insertion_point = np.array(insertion_point)
        
        try:
            # Get target level from fiducial name if not provided
            if target_level is None:
                # Try to get the level from the fiducial name in the scene
                if hasattr(slicer, 'mrmlScene'):
                    fidNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
                    for fidNode in fidNodes:
                        if fidNode.GetNumberOfControlPoints() > 0:
                            for i in range(fidNode.GetNumberOfControlPoints()):
                                pos = [0, 0, 0]
                                fidNode.GetNthControlPointPosition(i, pos)
                                # Check if this is the insertion point
                                if np.linalg.norm(np.array(pos) - self.insertion_point) < 5:  # Within 5mm
                                    # Get the level from the label
                                    label = fidNode.GetNthControlPointLabel(i)
                                    # Extract level from labels like "Fid1 - L4 - Right"
                                    parts = label.split(" - ")
                                    if len(parts) >= 2:
                                        target_level = parts[1].strip()
                                        self.logger.info(f"Using level {target_level} from fiducial {label}")
                                    break
            
            if target_level is None:
                self.logger.warning("No target level specified, will process all vertebrae")
            else:
                self.logger.info(f"Processing vertebra at level {target_level}")
            
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
                    self.volume_node = volume_node
            else:
                # Assume it's a vtkImageData or similar
                self.logger.warning("Volume node doesn't have GetClassName method")
                self.volume_node = volume_node
            
            # Convert nodes to numpy arrays for processing
            self.mask_array = self._volume_to_numpy_array(self.mask_node)
            self.logger.info(f"Mask array shape: {self.mask_array.shape}, non-zero elements: {np.count_nonzero(self.mask_array)}")
            
            self.volume_array = self._volume_to_numpy_array(self.volume_node)
            self.logger.info(f"Volume array shape: {self.volume_array.shape}")
            
            # Create masked volume (volume × mask)
            self.masked_volume_array = self.volume_array * (self.mask_array > 0).astype(float)
            
            # Make masked volume available to other components
            self.maskedVolume = self._numpy_to_volume(self.masked_volume_array, self.volume_node)
            if self.maskedVolume:
                self.maskedVolume.SetName("MaskedVertebraVolume")
            
            # Calculate centroid of the vertebra
            self.centroid = self._calculate_centroid_with_label(self.mask_array, target_level)
            self.logger.info(f"Vertebra centroid (RAS): {self.centroid}")
            
            # Extract surface for collision detection
            self.surface_array = self._extract_surface_voxels(self.mask_array)
            self.point_cloud = self._array_to_point_cloud(
                self.surface_array,
                self.volume_node,
                threshold=0
            )
            
            # REMOVED: Old pedicle detection using canal-based approach
            
            # NEW: Detect pedicle center and border using the smallest cross-section approach
            self.pedicle_center_point, self.pedicle_border_cloud, self.pcaVectors = \
                self.detect_pedicle_center_and_border(target_level)
            
            self.logger.info(f"Pedicle center point: {self.pedicle_center_point}")
            self.logger.info(f"PCA principal vectors: {self.pcaVectors}")
            
            # Create a fiducial for the pedicle center point
            if hasattr(slicer, 'mrmlScene') and self.pedicle_center_point is not None:
                fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PedicleCenterPoint")
                fiducial_node.AddFiducial(*self.pedicle_center_point)
                fiducial_node.SetNthFiducialLabel(0, "Pedicle Center")
                fiducial_node.CreateDefaultDisplayNodes()
                display_node = fiducial_node.GetDisplayNode()
                if display_node:
                    display_node.SetSelectedColor(0.0, 1.0, 0.0)  # Green
                    display_node.SetGlyphScale(3.0)
            
            # Visualize PCA axes
            self._visualize_pca_axes()
            
            # Clean up temporary node
            if temp_labelmap_node:
                slicer.mrmlScene.RemoveNode(temp_labelmap_node)
                    
        except Exception as e:
            self.logger.error(f"Error initializing Vertebra: {str(e)}")
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
            self.logger.warning("Using insertion point as pedicle center due to calculation failure")
    
    def _visualize_pca_axes(self):
        """
        Create visual models for PCA axes of both the whole vertebra and the pedicle,
        highlighting the best-aligned axis for the vertebra.
        """
        if not hasattr(slicer, 'mrmlScene'):
            return
        
        # Visualize whole vertebra PCA axes
        if hasattr(self, 'pcaVectors') and hasattr(self, 'vertebra_center_ras'):
            # Colors for the three axes
            axis_colors = [
                (1.0, 0.0, 0.0),  # Red - First axis
                (0.0, 1.0, 0.0),  # Green - Second axis
                (0.0, 0.0, 1.0)   # Blue - Third axis
            ]
            
            # Display length for axes
            display_length = 30.0  # mm for vertebra axes (longer for better visibility)
            
            # Get the best-aligned axis index (default to 0 if not available)
            best_axis_idx = getattr(self, 'best_aligned_pca_idx', 0)
            
            for i in range(3):
                axis_vector = self.pcaVectors[:, i]
                vector_length = np.linalg.norm(axis_vector)
                
                if vector_length < 1e-6:
                    continue
                    
                # Normalize and scale vector
                axis_vector = axis_vector / vector_length * display_length
                
                # Create line endpoints
                start_point = self.vertebra_center_ras - axis_vector/2
                end_point = self.vertebra_center_ras + axis_vector/2
                
                # Create line source
                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(start_point)
                line_source.SetPoint2(end_point)
                line_source.Update()
                
                # Create model node
                model_name = f"Vertebra_PCA_Axis_{i+1}"
                if i == best_axis_idx:
                    model_name = f"Vertebra_PCA_Axis_{i+1}_BestAligned"
                
                model_node = slicer.mrmlScene.GetFirstNodeByName(model_name)
                if model_node:
                    # If the node exists, remove it first to avoid errors
                    slicer.mrmlScene.RemoveNode(model_node)
                    
                # Create a new model node
                model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
                model_node.SetAndObservePolyData(line_source.GetOutput())
                
                # Create display node
                display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                model_node.SetAndObserveDisplayNodeID(display_node.GetID())
                
                # Set display properties
                # Highlight the best-aligned axis with a different appearance
                if i == best_axis_idx:
                    # Make the best-aligned axis yellow and thicker
                    display_node.SetColor(1.0, 1.0, 0.0)  # Yellow
                    display_node.SetLineWidth(5.0)  # Thicker line
                else:
                    display_node.SetColor(*axis_colors[i])
                    display_node.SetLineWidth(3.0)
        
        # Optionally visualize pedicle PCA axes if available and different from vertebra PCA
        if hasattr(self, 'pedicle_coeff') and hasattr(self, 'pedicle_center_point'):
            # Check if we need to visualize separate pedicle axes
            # For simplicity, we'll just check if center points are different
            same_centers = False
            if hasattr(self, 'vertebra_center_ras'):
                same_centers = np.allclose(self.vertebra_center_ras, self.pedicle_center_point, atol=5.0)
                
            if not same_centers:  # Only visualize if they're distinct
                # Use a shorter display length for pedicle axes to differentiate
                pedicle_display_length = 15.0  # mm
                
                # Different color scheme for pedicle axes
                pedicle_axis_colors = [
                    (1.0, 0.5, 0.5),  # Light red
                    (0.5, 1.0, 0.5),  # Light green
                    (0.5, 0.5, 1.0)   # Light blue
                ]
                
                # Calculate and scale the pedicle eigenvectors
                if hasattr(self, 'pedicle_latent'):
                    scaling_factor = np.sqrt(self.pedicle_latent) * 2
                    pedicle_vectors = self.pedicle_coeff * scaling_factor[:, np.newaxis]
                else:
                    # If latent values not available, just use normalized vectors
                    pedicle_vectors = np.copy(self.pedicle_coeff)
                
                for i in range(3):
                    axis_vector = pedicle_vectors[i]
                    vector_length = np.linalg.norm(axis_vector)
                    
                    if vector_length < 1e-6:
                        continue
                        
                    # Normalize and scale vector
                    axis_vector = axis_vector / vector_length * pedicle_display_length
                    
                    # Create line endpoints
                    start_point = self.pedicle_center_point - axis_vector/2
                    end_point = self.pedicle_center_point + axis_vector/2
                    
                    # Create line source
                    line_source = vtk.vtkLineSource()
                    line_source.SetPoint1(start_point)
                    line_source.SetPoint2(end_point)
                    line_source.Update()
                    
                    # Create model node
                    model_name = f"Pedicle_PCA_Axis_{i+1}"
                    
                    model_node = slicer.mrmlScene.GetFirstNodeByName(model_name)
                    if model_node:
                        # If the node exists, remove it first to avoid errors
                        slicer.mrmlScene.RemoveNode(model_node)
                        
                    # Create a new model node
                    model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
                    model_node.SetAndObservePolyData(line_source.GetOutput())
                    
                    # Create display node
                    display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                    model_node.SetAndObserveDisplayNodeID(display_node.GetID())
                    
                    # Set display properties
                    display_node.SetColor(*pedicle_axis_colors[i])
                    display_node.SetLineWidth(2.0)  # Thinner line than vertebra axes
    
    def _ras_to_ijk(self, ras_point):
        """
        Convert RAS coordinates to IJK coordinates.
        
        Parameters:
            ras_point: 3D point in RAS coordinates
            
        Returns:
            numpy.ndarray: Point in IJK coordinates
        """
        try:
            if not hasattr(self, 'volume_node') or not self.volume_node:
                return ras_point
                
            if not hasattr(self.volume_node, 'GetRASToIJKMatrix'):
                self.logger.warning("Volume node doesn't have GetRASToIJKMatrix method")
                return ras_point
                
            # Get RAS to IJK transformation matrix
            ras_to_ijk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(ras_to_ijk)
            
            # Convert to homogeneous coordinates
            ras_homogeneous = np.append(ras_point, 1.0)
            
            # Apply transformation
            ijk_homogeneous = [0, 0, 0, 1]
            ras_to_ijk.MultiplyPoint(ras_homogeneous, ijk_homogeneous)
            
            # Return IJK coordinates
            return np.array(ijk_homogeneous[0:3])
            
        except Exception as e:
            self.logger.error(f"Error in _ras_to_ijk: {str(e)}")
            return ras_point
    
    def _ijk_to_ras(self, ijk_point):
        """
        Convert IJK coordinates to RAS coordinates.
        
        Parameters:
            ijk_point: 3D point in IJK coordinates
            
        Returns:
            numpy.ndarray: Point in RAS coordinates
        """
        try:
            if not hasattr(self, 'volume_node') or not self.volume_node:
                return ijk_point
                
            if not hasattr(self.volume_node, 'GetIJKToRASMatrix'):
                self.logger.warning("Volume node doesn't have GetIJKToRASMatrix method")
                return ijk_point
                
            # Get IJK to RAS transformation matrix
            ijk_to_ras = vtk.vtkMatrix4x4()
            self.volume_node.GetIJKToRASMatrix(ijk_to_ras)
            
            # Convert to homogeneous coordinates
            ijk_homogeneous = np.append(ijk_point, 1.0)
            
            # Apply transformation
            ras_homogeneous = [0, 0, 0, 1]
            ijk_to_ras.MultiplyPoint(ijk_homogeneous, ras_homogeneous)
            
            # Return RAS coordinates
            return np.array(ras_homogeneous[0:3])
            
        except Exception as e:
            self.logger.error(f"Error in _ijk_to_ras: {str(e)}")
            return ijk_point

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
        try:
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
                # If reference node doesn't have image data, use numpy array dimensions
                img_vtk.SetDimensions(numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
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
                volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
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
                
        except Exception as e:
            self.logger.error(f"Error in _numpy_to_volume: {str(e)}")
            return None
    
    def _calculate_centroid(self, mask_array, threshold=0):
        """
        Calculate the centroid of non-zero voxels in a binary mask.
        
        Parameters:
            mask_array: 3D numpy array
            threshold: Minimum value to consider as part of the object
            
        Returns:
            numpy.ndarray: [x, y, z] coordinates of centroid
        """
        try:
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
            
            # Convert from IJK to RAS
            return self._ijk_to_ras(centroid_ijk)
        except Exception as e:
            self.logger.error(f"Error in _calculate_centroid: {str(e)}")
            return self.insertion_point
    
    def _cut_pedicle_side(self, volume_array, insertion_x, centroid_x):
        """
        Cut the volume to keep only the relevant side (left or right).
        
        Parameters:
            volume_array: 3D numpy array
            insertion_x: X coordinate of insertion point in RAS
            centroid_x: X coordinate of vertebra centroid in RAS
            
        Returns:
            numpy.ndarray: Cropped array containing only one side of the pedicle
        """
        try:
            # Create a copy to avoid modifying the original
            result = volume_array.copy()
            
            # Log the input parameters
            self.logger.info(f"Cutting pedicle side with insertion_x={insertion_x}, centroid_x={centroid_x}")
            
            # Convert coordinates to image space if needed
            # Convert insertion point to IJK
            insertion_point_ijk = self._ras_to_ijk(np.array([insertion_x, 0, 0]))
            insertion_x_ijk = insertion_point_ijk[0]
            
            # Convert centroid to IJK
            centroid_point_ijk = self._ras_to_ijk(np.array([centroid_x, 0, 0]))
            centroid_x_ijk = centroid_point_ijk[0]
            
            self.logger.info(f"In IJK space: insertion_x_ijk={insertion_x_ijk}, centroid_x_ijk={centroid_x_ijk}")
            
            # Round to integer for indexing
            centroid_idx = int(np.round(centroid_x_ijk))
            centroid_idx = max(0, min(centroid_idx, result.shape[0] - 1))
            
            # Cut based on which side the insertion point is on
            if insertion_x_ijk > centroid_x_ijk:
                # Keep right side
                self.logger.info(f"Keeping right side of pedicle (insertion point is to the right of centroid)")
                if centroid_idx > 0:
                    result[0:centroid_idx, :, :] = 0
            else:
                # Keep left side
                self.logger.info(f"Keeping left side of pedicle (insertion point is to the left of centroid)")
                if centroid_idx < result.shape[0]:
                    result[centroid_idx:, :, :] = 0
            
            # Check if we have a valid result
            non_zero_count = np.count_nonzero(result)
            self.logger.info(f"Pedicle side non-zero voxel count: {non_zero_count}")
            
            if non_zero_count == 0:
                self.logger.warning("Pedicle side cutting resulted in empty array, using original pedicle volume")
                # If cutting resulted in empty array, return the original
                return volume_array
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _cut_pedicle_side: {str(e)}")
            self.logger.error(traceback.format_exc())
            return volume_array  # Return original on error
    
    def _extract_surface_voxels(self, volume_array):
        """
        Extract the surface voxels of a binary volume.
        
        Parameters:
            volume_array: 3D numpy array
            
        Returns:
            numpy.ndarray: Binary mask of surface voxels
        """
        try:
            # Create binary mask
            binary_mask = (volume_array > 0).astype(np.uint8)
            
            # Erode the mask to find surface voxels
            from scipy.ndimage import binary_erosion
            
            # Use a 3D structuring element (6-connectivity)
            struct_elem = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            ])
            
            # Erode the mask
            eroded = binary_erosion(binary_mask, structure=struct_elem)
            
            # Surface is the difference between original and eroded
            surface = np.logical_and(binary_mask, np.logical_not(eroded))
            
            return surface.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in _extract_surface_voxels: {str(e)}")
            self.logger.error(traceback.format_exc())
            return (volume_array > 0).astype(np.uint8)  # Return binary mask on error
        
    def _extract_surface_voxels_with_label(self, volume_array, mask_array, target_level=None):
        """
        Extract the surface voxels of a binary volume, filtered by the target level.
        
        Parameters:
            volume_array: 3D numpy array of the volume
            mask_array: 3D numpy array of the segmentation mask
            target_level: Target level string (e.g., 'L4')
            
        Returns:
            numpy.ndarray: Binary mask of surface voxels for the target vertebra
        """
        try:
            # Find which label corresponds to the target level
            level_label = None
            
            # If we have a segmentation labelmap with proper labels
            if hasattr(self, 'mask_node') and hasattr(self.mask_node, 'GetLabelName'):
                # Try to find the label corresponding to the target level
                num_labels = self.mask_node.GetNumberOfLabels()
                for i in range(num_labels):
                    label_name = self.mask_node.GetLabelName(i)
                    if label_name and target_level and target_level in label_name:
                        level_label = i
                        self.logger.info(f"Found label {level_label} for {target_level}: {label_name}")
                        break
            
            # If we don't have label information, check if the mask has discrete values
            if level_label is None:
                # Get unique values in the mask (excluding 0)
                unique_values = np.unique(mask_array)
                unique_values = unique_values[unique_values > 0]
                
                # Map lumbar levels to typical label values if needed
                level_mapping = {
                    'L1': 5,
                    'L2': 4,
                    'L3': 3,
                    'L4': 2,
                    'L5': 1
                }
                
                # Try to find the label based on the mapping
                if target_level and target_level in level_mapping and level_mapping[target_level] in unique_values:
                    level_label = level_mapping[target_level]
                    self.logger.info(f"Using mapped label {level_label} for {target_level}")
                elif len(unique_values) == 1:
                    # If there's only one segmentation, use it
                    level_label = unique_values[0]
                    self.logger.info(f"Using only available label {level_label}")
                else:
                    # Can't determine which label to use
                    self.logger.warning(f"Cannot determine label for {target_level}, using all segmentations")
            
            # Create binary mask for the target vertebra
            if level_label is not None:
                self.logger.info(f"Creating binary mask for label {level_label}")
                binary_mask = (mask_array == level_label)
            else:
                # If we can't determine the label, use the entire segmentation
                self.logger.info("Creating binary mask for all segmentations")
                binary_mask = (mask_array > 0)
            
            # Log the number of voxels in the binary mask
            self.logger.info(f"Binary mask has {np.count_nonzero(binary_mask)} voxels")
            
            # Check if we have a valid mask
            if np.count_nonzero(binary_mask) == 0:
                self.logger.warning("Empty binary mask, cannot extract surface voxels")
                return np.zeros_like(binary_mask)
            
            # Use morphological operations to find surface voxels
            try:
                from scipy.ndimage import binary_erosion
                
                # Use a 3D 6-connectivity structuring element
                struct_elem = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
                
                # Erode the mask to get the interior
                eroded = binary_erosion(binary_mask, structure=struct_elem)
                
                # Surface voxels are those in the original mask but not in the eroded mask
                surface = np.logical_and(binary_mask, np.logical_not(eroded))
                
                # Check if we have a valid surface
                surface_count = np.count_nonzero(surface)
                self.logger.info(f"Extracted {surface_count} surface voxels")
                
                if surface_count == 0:
                    self.logger.warning("No surface voxels found after erosion, using original mask")
                    return binary_mask
                
                return surface.astype(np.uint8)
                
            except ImportError:
                # If scipy is not available, use a simpler approach
                self.logger.warning("scipy.ndimage not available, using simple dilation approach")
                
                # Simple 6-connectivity check
                surface = np.zeros_like(binary_mask)
                
                # Get dimensions
                nx, ny, nz = binary_mask.shape
                
                # Define 6-connectivity neighbors
                neighbors = [
                    (-1, 0, 0), (1, 0, 0),
                    (0, -1, 0), (0, 1, 0),
                    (0, 0, -1), (0, 0, 1)
                ]
                
                # For each voxel in the mask, check if it's on the surface
                # A voxel is on the surface if at least one of its 6-neighbors is background
                voxel_coords = np.where(binary_mask)
                for i in range(len(voxel_coords[0])):
                    x, y, z = voxel_coords[0][i], voxel_coords[1][i], voxel_coords[2][i]
                    
                    for dx, dy, dz in neighbors:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        # Check bounds
                        if (0 <= nx < binary_mask.shape[0] and 
                            0 <= ny < binary_mask.shape[1] and 
                            0 <= nz < binary_mask.shape[2]):
                            
                            # If this neighbor is background, the current voxel is on the surface
                            if binary_mask[nx, ny, nz] == 0:
                                surface[x, y, z] = 1
                                break
                
                return surface
                
        except Exception as e:
            self.logger.error(f"Error in _extract_surface_voxels_with_label: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return the binary mask as fallback
            return (mask_array > 0).astype(np.uint8)
    
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
        try:
            # Find indices of voxels above threshold
            indices = np.where(array > threshold)
            
            if len(indices[0]) == 0:
                # Return empty point cloud
                self.logger.warning("No points above threshold in array")
                points = vtk.vtkPoints()
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                return polydata
            
            # Create points
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(len(indices[0]))
            
            # For each voxel, convert IJK to RAS and add to points
            for i in range(len(indices[0])):
                # IJK coordinates
                ijk = np.array([indices[0][i], indices[1][i], indices[2][i]])
                
                # Convert to RAS
                ras = self._ijk_to_ras(ijk)
                
                # Add point
                points.SetPoint(i, ras[0], ras[1], ras[2])
            
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
            
        except Exception as e:
            self.logger.error(f"Error in _array_to_point_cloud: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return empty point cloud on error
            points = vtk.vtkPoints()
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            return polydata
        
    def _array_to_point_cloud_with_label(self, array, mask_array, target_level, reference_node=None, threshold=0):
        """
        Convert a binary array to a VTK point cloud, filtered by the target level.
        
        Parameters:
            array: 3D numpy array to extract points from
            mask_array: 3D numpy array of the segmentation mask
            target_level: Target level string (e.g., 'L4')
            reference_node: Volume node for coordinate transform
            threshold: Minimum value to include
            
        Returns:
            vtkPolyData: Point cloud of non-zero voxels from the target vertebra
        """
        try:
            # Find which label corresponds to the target level
            level_label = None
            
            # If we have a segmentation labelmap with proper labels
            if hasattr(self, 'mask_node') and hasattr(self.mask_node, 'GetLabelName'):
                # Try to find the label corresponding to the target level
                num_labels = self.mask_node.GetNumberOfLabels()
                for i in range(num_labels):
                    label_name = self.mask_node.GetLabelName(i)
                    if label_name and target_level and target_level in label_name:
                        level_label = i
                        self.logger.info(f"Found label {level_label} for {target_level}: {label_name}")
                        break
            
            # If we don't have label information, check if the mask has discrete values
            if level_label is None and target_level:
                # Get unique values in the mask (excluding 0)
                unique_values = np.unique(mask_array)
                unique_values = unique_values[unique_values > 0]
                
                # Map lumbar levels to typical label values if needed
                level_mapping = {
                    'L1': 5,
                    'L2': 4,
                    'L3': 3,
                    'L4': 2,
                    'L5': 1
                }
                
                # Try to find the label based on the mapping
                if target_level in level_mapping and level_mapping[target_level] in unique_values:
                    level_label = level_mapping[target_level]
                    self.logger.info(f"Using mapped label {level_label} for {target_level}")
                elif len(unique_values) == 1:
                    # If there's only one segmentation, use it
                    level_label = unique_values[0]
                    self.logger.info(f"Using only available label {level_label}")
                else:
                    # Can't determine which label to use
                    self.logger.warning(f"Cannot determine label for {target_level}, using all segmentations")
            
            # Create a mask for the specified vertebra
            label_mask = None
            if level_label is not None:
                label_mask = (mask_array == level_label)
                self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(label_mask)} voxels")
            else:
                # If we can't determine the label, use all non-zero voxels
                label_mask = (mask_array > 0)
                self.logger.info(f"Using all segmentations with {np.count_nonzero(label_mask)} voxels")
            
            # Apply the label mask to the input array
            filtered_array = array * label_mask
            
            # Find indices of voxels above threshold in the filtered array
            indices = np.where(filtered_array > threshold)
            
            if len(indices[0]) == 0:
                # No points found with the label filter, try using the unfiltered array
                self.logger.warning(f"No points found in filtered array, using original array")
                indices = np.where(array > threshold)
                
                if len(indices[0]) == 0:
                    # Return empty point cloud
                    self.logger.warning("No points above threshold in array")
                    points = vtk.vtkPoints()
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(points)
                    return polydata
            
            # Create points
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(len(indices[0]))
            
            # For each voxel, convert IJK to RAS and add to points
            for i in range(len(indices[0])):
                # IJK coordinates
                ijk = np.array([indices[0][i], indices[1][i], indices[2][i]])
                
                # Convert to RAS
                ras = self._ijk_to_ras(ijk)
                
                # Add point
                points.SetPoint(i, ras[0], ras[1], ras[2])
            
            # Create polydata with vertex cells
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            
            # Add vertices (improves rendering)
            vertices = vtk.vtkCellArray()
            for i in range(points.GetNumberOfPoints()):
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(i)
            
            polydata.SetVerts(vertices)
            
            # Log the number of points in the point cloud
            self.logger.info(f"Created point cloud with {points.GetNumberOfPoints()} points")
            
            return polydata
            
        except Exception as e:
            self.logger.error(f"Error in _array_to_point_cloud_with_label: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return empty point cloud on error
            points = vtk.vtkPoints()
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            return polydata
    
    def _points_to_numpy(self, vtk_points):
        """
        Convert vtkPoints to numpy array.
        
        Parameters:
            vtk_points: vtkPoints object
            
        Returns:
            numpy.ndarray: Array of point coordinates (n_points × 3)
        """
        try:
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
            except Exception as inner_e:
                self.logger.warning(f"Direct conversion failed: {str(inner_e)}")
            
            # Manual conversion as fallback
            result = np.zeros((n_points, 3))
            for i in range(n_points):
                result[i] = vtk_points.GetPoint(i)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _points_to_numpy: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.zeros((0, 3))
        
    def get_vertebra_label_mapping(self, segmentation_node):
        """
        Generate a mapping from vertebra names (e.g., 'L4') to their corresponding label values
        in the segmentation.
        
        Parameters:
            segmentation_node: vtkMRMLSegmentationNode containing the vertebra segments
            
        Returns:
            dict: Mapping from vertebra names to label values
        """
        try:
            if not segmentation_node:
                self.logger.warning("No segmentation node provided")
                return {}
                
            # Default mapping for standard vertebra names
            default_mapping = {
                "L1": 5,
                "L2": 4,
                "L3": 3,
                "L4": 2,
                "L5": 1,
                # Add more vertebra levels if needed
                "T12": 6,
                "S1": 0
            }
            
            # Try to create a mapping from segment names in the segmentation
            custom_mapping = {}
            
            # If it's a segmentation node with segments
            if hasattr(segmentation_node, 'GetSegmentation'):
                segmentation = segmentation_node.GetSegmentation()
                if segmentation:
                    # Iterate through all segments
                    for i in range(segmentation.GetNumberOfSegments()):
                        segment_id = segmentation.GetNthSegmentID(i)
                        segment = segmentation.GetSegment(segment_id)
                        if segment:
                            segment_name = segment.GetName()
                            
                            # Try to extract vertebra level from segment name
                            # Examples: "L4 vertebra", "vertebra L4", "L4", etc.
                            vertebra_patterns = [
                                r'([TLC][0-9]+[\-\s]?[0-9]*)[\s\-_]+vertebra',  # L4 vertebra, T12 vertebra
                                r'vertebra[\s\-_]+([TLC][0-9]+[\-\s]?[0-9]*)',  # vertebra L4, vertebra T12
                                r'^([TLC][0-9]+[\-\s]?[0-9]*)$',                # L4, T12
                                r'([TLC][0-9]+[\-\s]?[0-9]*)'                   # Any occurrence of L4, T12, etc.
                            ]
                            
                            import re
                            for pattern in vertebra_patterns:
                                match = re.search(pattern, segment_name)
                                if match:
                                    level = match.group(1).replace(" ", "").replace("-", "")
                                    custom_mapping[level] = i + 1  # Label values typically start at 1
                                    self.logger.info(f"Mapped '{level}' to label {i+1} from segment '{segment_name}'")
                                    break
            
            # If it's a labelmap node with label descriptions
            elif hasattr(segmentation_node, 'GetLabelName'):
                for i in range(segmentation_node.GetNumberOfLabels()):
                    label_name = segmentation_node.GetLabelName(i)
                    if label_name:
                        # Apply the same regex patterns
                        import re
                        for pattern in [
                            r'([TLC][0-9]+[\-\s]?[0-9]*)[\s\-_]+vertebra',
                            r'vertebra[\s\-_]+([TLC][0-9]+[\-\s]?[0-9]*)',
                            r'^([TLC][0-9]+[\-\s]?[0-9]*)$',
                            r'([TLC][0-9]+[\-\s]?[0-9]*)'
                        ]:
                            match = re.search(pattern, label_name)
                            if match:
                                level = match.group(1).replace(" ", "").replace("-", "")
                                custom_mapping[level] = i
                                self.logger.info(f"Mapped '{level}' to label {i} from label name '{label_name}'")
                                break
            
            # If we found custom mappings, use those; otherwise fall back to default
            if custom_mapping:
                self.logger.info(f"Using custom vertebra label mapping: {custom_mapping}")
                return custom_mapping
            else:
                self.logger.info(f"Using default vertebra label mapping: {default_mapping}")
                return default_mapping
                
        except Exception as e:
            self.logger.error(f"Error in get_vertebra_label_mapping: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return default mapping on error
            return {
                "L1": 5,
                "L2": 4,
                "L3": 3,
                "L4": 2,
                "L5": 1
            }
        
    def _calculate_centroid_with_label(self, mask_array, target_level=None, threshold=0):
        """
        Calculate the centroid of non-zero voxels in a binary mask for a specific vertebra level.
        
        Parameters:
            mask_array: 3D numpy array of the segmentation mask
            target_level: Target level string (e.g., 'L4')
            threshold: Minimum value to consider as part of the object
            
        Returns:
            numpy.ndarray: [x, y, z] coordinates of centroid
        """
        try:
            # If no target level specified, fall back to regular centroid calculation
            if target_level is None:
                return self._calculate_centroid(mask_array, threshold)
                
            # Find which label corresponds to the target level
            level_label = None
            
            # If we have a segmentation labelmap with proper labels
            if hasattr(self, 'mask_node') and hasattr(self.mask_node, 'GetLabelName'):
                # Try to find the label corresponding to the target level
                num_labels = self.mask_node.GetNumberOfLabels()
                for i in range(num_labels):
                    label_name = self.mask_node.GetLabelName(i)
                    if label_name and target_level in label_name:
                        level_label = i
                        self.logger.info(f"Found label {level_label} for {target_level}: {label_name}")
                        break
            
            # If we don't have label information, check if the mask has discrete values
            if level_label is None:
                # Get unique values in the mask (excluding 0)
                unique_values = np.unique(mask_array)
                unique_values = unique_values[unique_values > 0]
                
                # Map lumbar levels to typical label values if needed
                level_mapping = {
                    'L1': 5,
                    'L2': 4,
                    'L3': 3,
                    'L4': 2,
                    'L5': 1
                }
                
                # Try to find the label based on the mapping
                if target_level in level_mapping and level_mapping[target_level] in unique_values:
                    level_label = level_mapping[target_level]
                    self.logger.info(f"Using mapped label {level_label} for {target_level}")
                elif len(unique_values) == 1:
                    # If there's only one segmentation, use it
                    level_label = unique_values[0]
                    self.logger.info(f"Using only available label {level_label}")
                else:
                    # Can't determine which label to use
                    self.logger.warning(f"Cannot determine label for {target_level}, using all segmentations")
            
            # Create a mask for the specified vertebra
            vertebra_mask = None
            if level_label is not None:
                vertebra_mask = (mask_array == level_label)
                self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(vertebra_mask)} voxels")
                
                # If the mask is empty, fall back to using all non-zero voxels
                if np.count_nonzero(vertebra_mask) == 0:
                    self.logger.warning(f"Empty mask for label {level_label}, using all non-zero voxels")
                    vertebra_mask = (mask_array > threshold)
            else:
                # If we can't determine the label, use all non-zero voxels
                vertebra_mask = (mask_array > threshold)
                
            # Find indices of non-zero voxels in the filtered mask
            indices = np.where(vertebra_mask)
            
            if len(indices[0]) == 0:
                self.logger.warning("No voxels above threshold for centroid calculation")
                return self.insertion_point
            
            # Calculate mean position in IJK coordinates
            centroid_ijk = np.array([
                np.mean(indices[0]),
                np.mean(indices[1]),
                np.mean(indices[2])
            ])
            
            # Convert from IJK to RAS
            centroid_ras = self._ijk_to_ras(centroid_ijk)
            
            self.logger.info(f"Calculated centroid for {target_level}: IJK={centroid_ijk}, RAS={centroid_ras}")
            return centroid_ras
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_centroid_with_label: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Fall back to the insertion point in case of error
            return self.insertion_point
        
    def _numpy_to_vtk_polydata(self, points):
        """
        Convert numpy array of points to vtkPolyData.
        
        Parameters:
            points: Numpy array of points (n_points × 3)
            
        Returns:
            vtkPolyData: VTK polydata object containing the points
        """
        import vtk
        import numpy as np
        
        if points is None or len(points) == 0:
            # Return empty polydata
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk.vtkPoints())
            return polydata
        
        # Create vtkPoints object
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(len(points))
        
        # Set the points
        for i in range(len(points)):
            vtk_points.SetPoint(i, points[i][0], points[i][1], points[i][2])
        
        # Create vtkPolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        
        # Add vertices (improves rendering)
        vertices = vtk.vtkCellArray()
        for i in range(vtk_points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        
        polydata.SetVerts(vertices)
        
        return polydata
    
    def transform_to_aligned_space(self, points, center_of_mass, rotation_matrix):
        """
        Transform points from original space to aligned space.
        
        Parameters:
            points: Numpy array of points to transform
            center_of_mass: Center of mass used for alignment
            rotation_matrix: Rotation matrix used for alignment
            
        Returns:
            numpy.ndarray: Transformed points in aligned space
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        if points is None:
            return None
        
        # Convert to numpy array if it's not already
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Reshape to handle both single points and arrays
        original_shape = points.shape
        if len(original_shape) == 1:
            points = points.reshape(1, -1)
        
        # Create rotation object from matrix
        rotation = R.from_matrix(rotation_matrix)
        
        # Apply rotation
        centered_points = points - center_of_mass
        aligned_points = rotation.apply(centered_points) + center_of_mass
        
        # Restore original shape if it was a single point
        if len(original_shape) == 1:
            aligned_points = aligned_points[0]
        
        return aligned_points

    def transform_to_original_space(self, points, center_of_mass, rotation_matrix):
        """
        Transform points from aligned space back to original space.
        
        Parameters:
            points: Numpy array of points to transform
            center_of_mass: Center of mass used for alignment
            rotation_matrix: Rotation matrix used for alignment
            
        Returns:
            numpy.ndarray: Transformed points in original space
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        if points is None or len(points) == 0:
            return None
        
        # Create rotation object from matrix
        rotation = R.from_matrix(rotation_matrix)
        
        # Reshape for single point case
        original_shape = points.shape
        if len(original_shape) == 1:
            points = points.reshape(1, -1)
        
        # Apply inverse rotation
        centered_points = points - center_of_mass
        original_points = rotation.inv().apply(centered_points) + center_of_mass
        
        # Restore original shape if it was a single point
        if len(original_shape) == 1:
            original_points = original_points[0]
        
        return original_points
        
    def _get_level_specific_mask(self, target_level=None):
        """
        Get a binary mask for the specific vertebra level.
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
            
        Returns:
            numpy.ndarray: Binary mask for the specified level
        """
        import numpy as np
        
        # If no target level specified, use all non-zero voxels
        if target_level is None:
            return (self.mask_array > 0).astype(np.uint8)
        
        # Find which label corresponds to the target level
        level_label = None
        
        # If we have a segmentation labelmap with proper labels
        if hasattr(self, 'mask_node') and hasattr(self.mask_node, 'GetLabelName'):
            # Try to find the label corresponding to the target level
            num_labels = self.mask_node.GetNumberOfLabels()
            for i in range(num_labels):
                label_name = self.mask_node.GetLabelName(i)
                if label_name and target_level in label_name:
                    level_label = i
                    self.logger.info(f"Found label {level_label} for {target_level}: {label_name}")
                    break
        
        # If we don't have label information, check if the mask has discrete values
        if level_label is None:
            # Map lumbar levels to typical label values if needed
            level_mapping = {
                'L1': 5,
                'L2': 4,
                'L3': 3,
                'L4': 2,
                'L5': 1
            }
            
            # Get unique values in the mask (excluding 0)
            unique_values = np.unique(self.mask_array)
            unique_values = unique_values[unique_values > 0]
            
            # Try to find the label based on the mapping
            if target_level in level_mapping and level_mapping[target_level] in unique_values:
                level_label = level_mapping[target_level]
                self.logger.info(f"Using mapped label {level_label} for {target_level}")
            elif len(unique_values) == 1:
                # If there's only one segmentation, use it
                level_label = unique_values[0]
                self.logger.info(f"Using only available label {level_label}")
            else:
                # Can't determine which label to use
                self.logger.warning(f"Cannot determine label for {target_level}, using all segmentations")
                return (self.mask_array > 0).astype(np.uint8)
        
        # Create level-specific mask
        if level_label is not None:
            level_mask = (self.mask_array == level_label)
            self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(level_mask)} voxels")
            
            # If the mask is empty, fall back to using all segmentations
            if np.count_nonzero(level_mask) == 0:
                self.logger.warning(f"Empty mask for label {level_label}, using all segmentations")
                return (self.mask_array > 0).astype(np.uint8)
            
            return level_mask.astype(np.uint8)
        
        # Default: use all non-zero voxels
        return (self.mask_array > 0).astype(np.uint8)
    
    def extract_volume_and_surface_clouds(self, target_level=None):
        """
        Extract volume and surface point clouds for a specific vertebra level.
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
            
        Returns:
            tuple: (volume_cloud, surface_cloud, level_mask) - vtk.vtkPolyData objects and binary mask
        """
        # First, get the level-specific mask
        level_mask = self._get_level_specific_mask(target_level)
        
        # Extract volume point cloud
        volume_cloud = self._array_to_point_cloud(
            self.volume_array * level_mask,
            self.volume_node,
            threshold=0
        )
        
        # Extract surface point cloud
        surface_voxels = self._extract_surface_voxels(level_mask)
        surface_cloud = self._array_to_point_cloud(
            surface_voxels,
            self.volume_node,
            threshold=0
        )
        
        return volume_cloud, surface_cloud, level_mask
    
    def select_best_alignment_vector(self, coeff, latent):
        """
        Select the best PCA vector for alignment based on anatomical directions.
        
        The best vector for alignment should be the one that most closely
        corresponds to the anterior-posterior direction of the vertebra.
        
        Parameters:
            coeff: PCA coefficient matrix (eigenvectors)
            latent: Eigenvalues from PCA
            
        Returns:
            tuple: (best_vector_idx, best_vector, alignment_confidence)
        """
        import numpy as np
        
        # RAS coordinate system: x=Right, y=Anterior, z=Superior
        # In the vertebral anatomy, the primary axis usually runs superior-inferior,
        # and we want to align with the anterior-posterior axis
        
        # Normalized anatomical direction vectors in RAS coordinates
        ap_axis = np.array([0, 1, 0])  # Anterior-posterior (y-axis)
        si_axis = np.array([0, 0, 1])  # Superior-inferior (z-axis)
        rl_axis = np.array([1, 0, 0])  # Right-left (x-axis)
        
        # Calculate alignment scores for each PCA vector
        alignment_scores = []
        
        for i in range(len(coeff)):
            # Normalize the eigenvector
            vector = coeff[i] / np.linalg.norm(coeff[i])
            
            # Calculate absolute dot products with anatomical axes
            # Higher values mean better alignment
            ap_score = abs(np.dot(vector, ap_axis))
            si_score = abs(np.dot(vector, si_axis))
            rl_score = abs(np.dot(vector, rl_axis))
            
            # We prefer vectors that align with anterior-posterior axis
            # but normalize by eigenvalue to consider variance importance
            normalized_eigenvalue = latent[i] / sum(latent)
            alignment_score = ap_score * normalized_eigenvalue
            
            alignment_scores.append((i, vector, alignment_score))
        
        # Sort by alignment score (highest first)
        alignment_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Return the best match
        best_idx, best_vector, confidence = alignment_scores[0]
        
        self.logger.info(f"Selected PCA vector {best_idx} as best alignment vector")
        self.logger.info(f"Alignment confidence: {confidence:.4f}")
        self.logger.info(f"Best alignment vector: {best_vector}")
        
        return best_idx, best_vector, confidence
    
    def align_point_cloud_with_pca(self, volume_cloud, surface_cloud, level_mask=None):
        """
        Align point clouds with the principal component analysis vectors.
        
        Parameters:
            volume_cloud: vtkPolyData of the volume point cloud
            surface_cloud: vtkPolyData of the surface point cloud
            level_mask: Binary mask for the specific level (optional)
            
        Returns:
            tuple: (aligned_volume_cloud, aligned_surface_cloud, rotation_matrix, 
                    center_of_mass, aligned_center, pca_vectors)
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Convert vtkPolyData to numpy array
        volume_points = self._points_to_numpy(volume_cloud.GetPoints())
        
        # Skip if we don't have enough points
        if volume_points.shape[0] < 10:
            self.logger.warning("Not enough points for PCA")
            return volume_cloud, surface_cloud, np.eye(3), np.zeros(3), np.zeros(3), np.eye(3)
        
        # Apply PCA
        coeff, latent, score = apply_pca(volume_points)
        
        # Get the center of mass
        center_of_mass = np.mean(volume_points, axis=0)
        
        # Select the best vector for alignment
        _, main_axis, _ = self.select_best_alignment_vector(coeff, latent)
        
        # Target axis for alignment (anterior-posterior is Y axis in RAS)
        target_axis = np.array([0, 1, 0])
        
        # Create rotation matrix to align main_axis with target_axis
        rotation = R.align_vectors([target_axis], [main_axis])[0]
        rotation_matrix = rotation.as_matrix()
        
        # Transform volume points
        centered_volume_points = volume_points - center_of_mass
        aligned_volume_points = rotation.apply(centered_volume_points) + center_of_mass
        
        # Create new vtkPolyData for aligned volume
        aligned_volume_cloud = self._numpy_to_vtk_polydata(aligned_volume_points)
        
        # Transform surface points if available
        aligned_surface_cloud = None
        if surface_cloud and surface_cloud.GetNumberOfPoints() > 0:
            surface_points = self._points_to_numpy(surface_cloud.GetPoints())
            centered_surface_points = surface_points - center_of_mass
            aligned_surface_points = rotation.apply(centered_surface_points) + center_of_mass
            aligned_surface_cloud = self._numpy_to_vtk_polydata(aligned_surface_points)
        
        # Calculate aligned center
        aligned_center = np.mean(aligned_volume_points, axis=0)
        
        # Scale PCA vectors by eigenvalues for visualization
        pca_vectors = coeff * np.sqrt(latent)[:, np.newaxis]
        
        return aligned_volume_cloud, aligned_surface_cloud, rotation_matrix, center_of_mass, aligned_center, pca_vectors
    
    def find_smallest_pedicle_cross_section(self, aligned_volume_cloud, aligned_surface_cloud, aligned_center, center_of_mass, rotation_matrix, insertion_x):
        """
        Find the smallest cross-section of the pedicle by:
        1. First locating the spinal canal using the full vertebra
        2. Then finding the smallest cross-section on the side of interest
        
        Parameters:
            aligned_volume_cloud: vtkPolyData of the aligned volume point cloud
            aligned_surface_cloud: vtkPolyData of the aligned surface point cloud
            aligned_center: 3D array of the aligned centroid coordinates
            center_of_mass: Center of mass used for the alignment
            rotation_matrix: Rotation matrix used for the alignment
            insertion_x: X coordinate of insertion point in original space
            
        Returns:
            tuple: (min_slice_idx, min_slice_points, min_area) - Index, points, and area of smallest slice
        """
        import numpy as np
        from scipy.spatial import cKDTree
        import time
        
        start_time = time.time()
        self.logger.info("Starting smallest pedicle cross-section detection")
        
        # 1. Convert vtkPolyData to numpy array (use ALL points)
        volume_points = self._points_to_numpy(aligned_volume_cloud.GetPoints())
        
        if volume_points.shape[0] == 0:
            self.logger.warning("Empty point cloud, cannot find smallest cross-section")
            return 0, None, float('inf')
        
        # 2. Transform insertion point to aligned space to determine side
        insertion_point_aligned = self.transform_to_aligned_space(
            self.insertion_point, center_of_mass, rotation_matrix)
        
        # Determine side of interest (left or right based on insertion point)
        is_left_side = insertion_point_aligned[0] < aligned_center[0]
        side_text = "left" if is_left_side else "right"
        self.logger.info(f"Working on {side_text} side based on insertion point at {insertion_point_aligned}")
        
        # 3. Build KD-tree for the FULL point cloud (don't filter by side yet)
        full_tree = cKDTree(volume_points)
        
        # Parameters for edge detection - these might need tuning
        sphere_radius = 1.0      # Search radius in mm
        point_threshold = 1      # Minimum points to consider as "edge"
        step_size = 0.1            # Step size for movement in mm
        search_range = 40.0      # Maximum distance to search for edges
        
        # 4. Create a test point at the center, looking for the spinal canal
        test_x = aligned_center[0]  # Use full center, not side-specific offset
        start_y = aligned_center[1]
        test_z = aligned_center[2]
        
        # Try to find a point in the void by sampling around the aligned center
        test_point = np.array([test_x, start_y, test_z])
        self.logger.info(f"Starting test point: {test_point}")
        
        # 5. Check if starting point is in the void (spinal canal)
        indices = full_tree.query_ball_point(test_point, sphere_radius)
        in_void = len(indices) < point_threshold
        
        # If not in void, search for the void around the center area
        if not in_void:
            self.logger.info("Starting point is not in the void, searching for the spinal canal...")
            
            # Try different y positions to find the void
            y_offsets = list(range(-30, 31, 1))  # Try ±30mm in 5mm steps
            
            candidate_points = []
            for y_offset in y_offsets:
                test_point[1] = start_y + y_offset
                indices = full_tree.query_ball_point(test_point, sphere_radius)
                
                self.logger.info(f"Testing y={test_point[1]}, found {len(indices)} points in sphere")
                
                if len(indices) < point_threshold:
                    in_void = True
                    self.logger.info(f"Found void at y={start_y}")
                    candidate_points.append(test_point[1])
            self.logger.info(f"Found {len(candidate_points)} points")

            # Search for the starting point that is closest to the aligned centroid
            min_so_far = candidate_points[0]
            for points in candidate_points:
                if np.abs(aligned_center[1] - points) < np.abs(aligned_center[1] - min_so_far):
                    min_so_far = points
                    self.logger.info(f"Minimum so far is at y={min_so_far} and the original points is y={aligned_center[1]}")
            
            start_y = min_so_far
            self.logger.info(f"Closest point is at y={min_so_far} and the original points is y={aligned_center[1]}")
            
            # If still not in void, try with different x offsets (on both sides)
            if not in_void:
                self.logger.info("Trying different x offsets for whole vertebra search...")
                x_offsets = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
                
                for x_offset in x_offsets:
                    test_x = aligned_center[0] + x_offset
                    for y_offset in y_offsets:
                        test_point = np.array([test_x, start_y + y_offset, test_z])
                        indices = full_tree.query_ball_point(test_point, sphere_radius)
                        
                        if len(indices) < point_threshold:
                            start_y = test_point[1]
                            test_x = test_point[0]
                            in_void = True
                            self.logger.info(f"Found void at x={test_x}, y={start_y}")
                            break
                    
                    if in_void:
                        break
        
        if not in_void:
            self.logger.warning("Could not find void (spinal canal), using centroid as starting point")
            # We'll continue but results may not be optimal
        
        # Reset test point with the found void location
        test_point = np.array([test_x, start_y, test_z])
        
        # Create debug markers for the search process
        if hasattr(slicer, 'mrmlScene'):
            # Remove existing node if it exists
            existing_node = slicer.mrmlScene.GetFirstNodeByName("PedicleVoidPoints")
            if existing_node:
                slicer.mrmlScene.RemoveNode(existing_node)
                
            debug_fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PedicleVoidPoints")
            debug_fiducials.CreateDefaultDisplayNodes()
            debug_fiducials.GetDisplayNode().SetSelectedColor(0.0, 1.0, 0.0)  # Green
            
            # Add starting void point
            idx = debug_fiducials.AddFiducial(test_point[0], test_point[1], test_point[2])
            debug_fiducials.SetNthFiducialLabel(idx, "Start Void")
        
        # 6. Now detect the anterior and posterior edges of the whole vertebra
        self.logger.info("Detecting spinal canal edges using FULL vertebra...")
        
        # Now we should be in the void or close to it, move anteriorly until hitting the vertebra
        anterior_edge = start_y
        anterior_found = False
        
        self.logger.info(f"Starting anterior search from y={anterior_edge}")
        max_y = start_y + search_range
        min_y = start_y - search_range
        
        # Search anteriorly (increasing y)
        current_point = np.array(test_point)
        for i in range(int(search_range / step_size)):
            current_point[1] += step_size
            
            # Check if we're beyond bounds
            if current_point[1] > max_y:
                self.logger.warning(f"Reached maximum y ({max_y}) without finding anterior edge")
                break
                
            # Check if we hit the vertebra
            indices = full_tree.query_ball_point(current_point, sphere_radius)
            num_points = len(indices)
            
            self.logger.debug(f"Anterior search at y={current_point[1]}, found {num_points} points")
            
            if num_points >= point_threshold:
                anterior_edge = current_point[1]
                anterior_found = True
                self.logger.info(f"Hit anterior edge at y={anterior_edge}")
                
                # Add debug marker
                if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                    idx = debug_fiducials.AddFiducial(current_point[0], current_point[1], current_point[2])
                    debug_fiducials.SetNthFiducialLabel(idx, "Anterior Edge")
                
                break
        
        # If we didn't find the anterior edge, use a default
        if not anterior_found:
            anterior_edge = np.percentile(volume_points[:, 1], 75)  # Use 75th percentile as fallback
            self.logger.warning(f"Using default anterior edge at y={anterior_edge}")
        
        # Move posteriorly until hitting the vertebra
        posterior_edge = start_y
        posterior_found = False
        
        self.logger.info(f"Starting posterior search from y={posterior_edge}")
        
        # Reset current point for posterior search
        current_point = np.array(test_point)
        for i in range(int(search_range / step_size)):
            current_point[1] -= step_size
            
            # Check if we're beyond bounds
            if current_point[1] < min_y:
                self.logger.warning(f"Reached minimum y ({min_y}) without finding posterior edge")
                break
                
            # Check if we hit the vertebra
            indices = full_tree.query_ball_point(current_point, sphere_radius)
            num_points = len(indices)
            
            self.logger.debug(f"Posterior search at y={current_point[1]}, found {num_points} points")
            
            if num_points >= point_threshold:
                posterior_edge = current_point[1]
                posterior_found = True
                self.logger.info(f"Hit posterior edge at y={posterior_edge}")
                
                # Add debug marker
                if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                    idx = debug_fiducials.AddFiducial(current_point[0], current_point[1], current_point[2])
                    debug_fiducials.SetNthFiducialLabel(idx, "Posterior Edge")
                
                break
        
        # If we didn't find the posterior edge, use a default
        if not posterior_found:
            posterior_edge = np.percentile(volume_points[:, 1], 25)  # Use 25th percentile as fallback
            self.logger.warning(f"Using default posterior edge at y={posterior_edge}")
        
        # Check if the anterior and posterior edges make sense
        if posterior_edge >= anterior_edge:
            self.logger.warning("Invalid pedicle edges detected, using statistical bounds")
            posterior_edge = np.percentile(volume_points[:, 1], 25)
            anterior_edge = np.percentile(volume_points[:, 1], 75)
            self.logger.info(f"Using statistical bounds: posterior={posterior_edge}, anterior={anterior_edge}")
        
        # 7. NOW filter for side of interest
        side_filter = volume_points[:, 0] < aligned_center[0] if is_left_side else volume_points[:, 0] > aligned_center[0]
        side_points = volume_points[side_filter]
        
        if len(side_points) == 0:
            self.logger.warning(f"No points found on the {side_text} side")
            return 0, None, float('inf')
        
        # Build KD-tree for side-specific search
        side_tree = cKDTree(side_points)
        
        # 8. Search for smallest cross-section within the range on the specific side
        min_area = float('inf')
        min_slice_idx = -1
        min_slice_points = None
        
        # Define slice thickness
        slice_thickness = 2.0  # mm
        
        # Search for the smallest cross-section (just in the side of interest)
        self.logger.info(f"Searching for smallest cross-section between y={posterior_edge} and y={anterior_edge}")
        
        # Create visualization for all slices (if in Slicer)
        cross_section_nodes = []
        
        for y_idx in np.arange(posterior_edge, anterior_edge, slice_thickness):
            # Find points in this slice
            slice_mask = (np.abs(side_points[:, 1] - y_idx) < slice_thickness/2)
            slice_points = side_points[slice_mask]
            
            # Calculate slice area (number of points as proxy for area)
            slice_area = len(slice_points)
            
            # Skip empty slices
            if slice_area == 0:
                continue
            
            # If we have points and area is smaller than current minimum
            if slice_area < min_area:
                min_area = slice_area
                min_slice_idx = y_idx
                min_slice_points = slice_points
                self.logger.debug(f"New minimum at y={y_idx}, area={slice_area}")
                
                # Visualize the current minimum slice if in Slicer
                if hasattr(slicer, 'mrmlScene') and slice_area > 0:
                    slice_cloud = self._numpy_to_vtk_polydata(slice_points)
                    slice_model_name = f"CrossSection_y{y_idx:.1f}"
                    
                    # Clean up old nodes with similar names
                    for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode"):
                        if node.GetName().startswith("CrossSection_") and node.GetName() not in cross_section_nodes:
                            slicer.mrmlScene.RemoveNode(node)
                    
                    cross_section_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", slice_model_name)
                    cross_section_model.SetAndObservePolyData(slice_cloud)
                    cross_section_model.CreateDefaultDisplayNodes()
                    if cross_section_model.GetDisplayNode():
                        # Progression from red to blue
                        progress = (y_idx - posterior_edge) / (anterior_edge - posterior_edge)
                        r = max(0, min(1, 2 * (1 - progress)))
                        b = max(0, min(1, 2 * progress - 1))
                        cross_section_model.GetDisplayNode().SetColor(r, 0.0, b)
                        cross_section_model.GetDisplayNode().SetOpacity(0.5)
                    
                    cross_section_nodes.append(slice_model_name)
        
        if min_slice_points is None or len(min_slice_points) == 0:
            self.logger.warning("No minimum slice found within pedicle boundaries")
            return 0, None, float('inf')
        
        self.logger.info(f"Found smallest cross-section at y={min_slice_idx} with area={min_area}")
        self.logger.info(f"Smallest cross-section detection took {time.time() - start_time:.2f} seconds")
        
        # Add marker for the smallest slice position (make it bigger and distinctive)
        if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
            idx = debug_fiducials.AddFiducial(
                test_x - (15 if is_left_side else -15),  # Offset to side of interest
                min_slice_idx, 
                test_z
            )
            debug_fiducials.SetNthFiducialLabel(idx, "Smallest Slice")
            debug_fiducials.SetNthFiducialSelected(idx, True)
            markup_display = debug_fiducials.GetDisplayNode()
            if markup_display:
                markup_display.SetGlyphScale(3.0)
                markup_display.SetSelectedColor(1.0, 0.0, 1.0)  # Magenta for the smallest slice
        
        # Create a specific model for the minimum slice (with different color)
        if hasattr(slicer, 'mrmlScene') and min_slice_points is not None and len(min_slice_points) > 0:
            min_slice_cloud = self._numpy_to_vtk_polydata(min_slice_points)
            min_slice_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "SmallestCrossSection")
            min_slice_model.SetAndObservePolyData(min_slice_cloud)
            min_slice_model.CreateDefaultDisplayNodes()
            if min_slice_model.GetDisplayNode():
                min_slice_model.GetDisplayNode().SetColor(1.0, 1.0, 0.0)  # Yellow
                min_slice_model.GetDisplayNode().SetOpacity(1.0)
        
        return min_slice_idx, min_slice_points, min_area
    
    def compute_pedicle_center_and_border(self, min_slice_points, aligned_surface_cloud, min_slice_idx, slice_thickness=2.0):
        """
        Compute the center of the pedicle and the pedicle border from the smallest cross-section.
        
        Parameters:
            min_slice_points: Numpy array of points in the smallest cross-section
            aligned_surface_cloud: vtkPolyData of the aligned surface point cloud
            min_slice_idx: Y-index of the smallest cross-section
            slice_thickness: Thickness of the slice to consider (mm)
            
        Returns:
            tuple: (pedicle_center, pedicle_border_points) - Center point and border points of pedicle
        """
        import numpy as np
        import time
        
        start_time = time.time()
        self.logger.info("Computing pedicle center and border...")
        
        # If we don't have any points, return default values
        if min_slice_points is None or len(min_slice_points) == 0:
            self.logger.warning("No points in smallest cross-section")
            return np.zeros(3), None
        
        # Compute the center of the smallest cross-section
        pedicle_center = np.mean(min_slice_points, axis=0)
        self.logger.info(f"Computed pedicle center at {pedicle_center}")
        
        # Extract the pedicle border from the surface cloud
        pedicle_border_points = None
        if aligned_surface_cloud and aligned_surface_cloud.GetNumberOfPoints() > 0:
            surface_points = self._points_to_numpy(aligned_surface_cloud.GetPoints())
            
            # Find surface points near the slice
            border_mask = np.abs(surface_points[:, 1] - min_slice_idx) < slice_thickness/2
            pedicle_border_points = surface_points[border_mask]
            
            self.logger.info(f"Found {len(pedicle_border_points) if pedicle_border_points is not None else 0} points on pedicle border")
            
            # Visualize the border points if in Slicer
            if hasattr(slicer, 'mrmlScene') and pedicle_border_points is not None and len(pedicle_border_points) > 0:
                # Remove existing border node if present
                existing_node = slicer.mrmlScene.GetFirstNodeByName("PedicleBorderSlice")
                if existing_node:
                    slicer.mrmlScene.RemoveNode(existing_node)
                    
                border_cloud = self._numpy_to_vtk_polydata(pedicle_border_points)
                border_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "PedicleBorderSlice")
                border_model.SetAndObservePolyData(border_cloud)
                border_model.CreateDefaultDisplayNodes()
                if border_model.GetDisplayNode():
                    border_model.GetDisplayNode().SetColor(0.0, 1.0, 1.0)  # Cyan
                    border_model.GetDisplayNode().SetPointSize(4.0)
        else:
            self.logger.warning("No surface cloud available to extract border")
        
        self.logger.info(f"Pedicle center and border computation took {time.time() - start_time:.2f} seconds")
        
        return pedicle_center, pedicle_border_points
    
    def detect_pedicle_center_and_border(self, target_level=None):
        """
        Detect the pedicle center and border using the smallest cross-section approach.
        This improved version:
        1. Uses the full vertebra to detect spinal canal
        2. Only filters by side after canal detection
        3. Finds the smallest cross-section of the pedicle
        4. Returns the pedicle center and border
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
            
        Returns:
            tuple: (pedicle_center, pedicle_border, pca_vectors) in original space
        """
        import numpy as np
        import time
        
        start_time = time.time()
        self.logger.info(f"Starting pedicle detection for level {target_level}")
        
        try:
            # Step 1: Extract volume and surface point clouds for the target level
            self.logger.info("Extracting point clouds...")
            volume_cloud, surface_cloud, level_mask = self.extract_volume_and_surface_clouds(target_level)
            
            if volume_cloud is None or volume_cloud.GetNumberOfPoints() == 0:
                self.logger.error("Failed to extract volume point cloud")
                return self.insertion_point, None, np.eye(3)
            
            # Step 2-3: Apply PCA and align point clouds
            self.logger.info("Applying PCA and aligning point clouds...")
            aligned_volume_cloud, aligned_surface_cloud, rotation_matrix, center_of_mass, aligned_center, pca_vectors = \
                self.align_point_cloud_with_pca(volume_cloud, surface_cloud, level_mask)
                
            # Store for visualization
            self.vertebra_center_ras = self._ijk_to_ras(center_of_mass) if hasattr(self, '_ijk_to_ras') else center_of_mass
            
            # Store best aligned PCA axis and score
            best_idx, _, best_score = self.select_best_alignment_vector(pca_vectors, np.array([0, 1, 0]))
            self.best_aligned_pca_idx = best_idx
            self.pca_alignment_score = best_score
            
            # VISUALIZATION: Create visualization of aligned vertebra
            self._visualize_aligned_vertebra(aligned_volume_cloud, aligned_surface_cloud, aligned_center, pca_vectors)
            
            # Step 4: Find smallest cross-section without pre-filtering by side
            self.logger.info("Finding smallest cross-section...")
            min_slice_idx, min_slice_points, min_area = self.find_smallest_pedicle_cross_section(
                aligned_volume_cloud, aligned_surface_cloud, aligned_center, 
                center_of_mass, rotation_matrix, self.insertion_point[0])
            
            # Log PCA and alignment details for debugging
            self.logger.info(f"PCA vectors shape: {pca_vectors.shape}")
            self.logger.info(f"Best aligned PCA axis: {best_idx} with score {best_score}")
            self.logger.info(f"Center of mass: {center_of_mass}")
            self.logger.info(f"Aligned center: {aligned_center}")
            
            # Step 5-6: Find pedicle center and border
            self.logger.info("Computing pedicle center and border...")
            aligned_pedicle_center, aligned_pedicle_border = self.compute_pedicle_center_and_border(
                min_slice_points, aligned_surface_cloud, min_slice_idx)
            
            # Step 7: Transform back to original space
            self.logger.info("Transforming back to original space...")
            pedicle_center = self.transform_to_original_space(
                aligned_pedicle_center, center_of_mass, rotation_matrix)
            
            # Create vtk polydata for pedicle border
            pedicle_border_cloud = None
            if aligned_pedicle_border is not None and len(aligned_pedicle_border) > 0:
                # Transform border points back to original space
                original_border_points = self.transform_to_original_space(
                    aligned_pedicle_border, center_of_mass, rotation_matrix)
                
                # Create vtk polydata
                pedicle_border_cloud = self._numpy_to_vtk_polydata(original_border_points)
            
            # Visualize the results if in Slicer environment
            if hasattr(slicer, 'mrmlScene'):
                # Clean up previous visualization nodes
                for name in ["PedicleCenter", "PedicleBorder"]:
                    existing_node = slicer.mrmlScene.GetFirstNodeByName(name)
                    if existing_node:
                        slicer.mrmlScene.RemoveNode(existing_node)
                        
                # Add fiducial for pedicle center
                fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PedicleCenter")
                fiducial_node.AddFiducial(*pedicle_center)
                fiducial_node.SetNthFiducialLabel(0, f"Pedicle Center {target_level}")
                fiducial_node.CreateDefaultDisplayNodes()
                if fiducial_node.GetDisplayNode():
                    fiducial_node.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)  # Orange
                    fiducial_node.GetDisplayNode().SetGlyphScale(3.0)
                
                # Visualize the pedicle border
                if pedicle_border_cloud and pedicle_border_cloud.GetNumberOfPoints() > 0:
                    border_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "PedicleBorder")
                    border_model.SetAndObservePolyData(pedicle_border_cloud)
                    border_model.CreateDefaultDisplayNodes()
                    if border_model.GetDisplayNode():
                        border_model.GetDisplayNode().SetColor(0.0, 1.0, 1.0)  # Cyan
                        border_model.GetDisplayNode().SetPointSize(4.0)
                
                # Create 3D model showing PCA axes
                self._visualize_pca_axes()
            
            self.logger.info(f"Pedicle detection completed in {time.time() - start_time:.2f} seconds")
            
            return pedicle_center, pedicle_border_cloud, pca_vectors
        
        except Exception as e:
            self.logger.error(f"Error in detect_pedicle_center_and_border: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return defaults on error
            return self.insertion_point, None, np.eye(3)

    def _visualize_aligned_vertebra(self, aligned_volume_cloud, aligned_surface_cloud, aligned_center, pca_vectors):
        """
        Visualize the aligned vertebra for debugging.
        
        Parameters:
            aligned_volume_cloud: vtkPolyData of the aligned volume point cloud
            aligned_surface_cloud: vtkPolyData of the aligned surface point cloud
            aligned_center: 3D coordinate of the aligned center point
            pca_vectors: Principal component vectors
        """
        import slicer
        import vtk
        import numpy as np
        
        # Only proceed if in Slicer environment
        if not hasattr(slicer, 'mrmlScene'):
            return
        
        # Remove any existing visualization nodes
        for name in ["AlignedVolume", "AlignedSurface", "AlignedCenterPoint", 
                    "AlignedAxis1", "AlignedAxis2", "AlignedAxis3"]:
            existing_node = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing_node:
                slicer.mrmlScene.RemoveNode(existing_node)
        
        # Create a model for the aligned volume point cloud
        if aligned_volume_cloud and aligned_volume_cloud.GetNumberOfPoints() > 0:
            volume_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "AlignedVolume")
            volume_model.SetAndObservePolyData(aligned_volume_cloud)
            volume_model.CreateDefaultDisplayNodes()
            if volume_model.GetDisplayNode():
                volume_model.GetDisplayNode().SetColor(0.7, 0.7, 0.9)  # Light blue
                volume_model.GetDisplayNode().SetOpacity(0.5)
                volume_model.GetDisplayNode().SetVisibility(0)  # Initially hidden
        
        # Create a model for the aligned surface point cloud
        if aligned_surface_cloud and aligned_surface_cloud.GetNumberOfPoints() > 0:
            surface_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "AlignedSurface")
            surface_model.SetAndObservePolyData(aligned_surface_cloud)
            surface_model.CreateDefaultDisplayNodes()
            if surface_model.GetDisplayNode():
                surface_model.GetDisplayNode().SetColor(0.9, 0.7, 0.7)  # Light red
                surface_model.GetDisplayNode().SetOpacity(0.8)
                surface_model.GetDisplayNode().SetVisibility(0)  # Initially hidden
        
        # Create a fiducial for the aligned center
        center_fiducial = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "AlignedCenterPoint")
        center_fiducial.AddFiducial(*aligned_center)
        center_fiducial.SetNthFiducialLabel(0, "Aligned Center")
        center_fiducial.CreateDefaultDisplayNodes()
        if center_fiducial.GetDisplayNode():
            center_fiducial.GetDisplayNode().SetSelectedColor(0.0, 1.0, 0.0)  # Green
            center_fiducial.GetDisplayNode().SetGlyphScale(5.0)
            center_fiducial.GetDisplayNode().SetVisibility(0)  # Initially hidden
        
        # Create visualization of PCA axes
        axis_colors = [
            (1.0, 0.0, 0.0),  # Red for first PC
            (0.0, 1.0, 0.0),  # Green for second PC
            (0.0, 0.0, 1.0)   # Blue for third PC
        ]
        
        axis_scale = 30.0  # Length of the axes
        
        for i in range(3):
            if pca_vectors.shape[1] <= i:
                continue
                
            vector = pca_vectors[:, i]
            norm = np.linalg.norm(vector)
            
            if norm < 1e-6:
                continue
                
            # Normalize and scale
            vector = vector / norm * axis_scale
            
            # Create line source for axis
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(aligned_center)
            line_source.SetPoint2(aligned_center + vector)
            line_source.Update()
            
            # Create model node
            axis_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"AlignedAxis{i+1}")
            axis_model.SetAndObservePolyData(line_source.GetOutput())
            axis_model.CreateDefaultDisplayNodes()
            
            # Set display properties
            if axis_model.GetDisplayNode():
                axis_model.GetDisplayNode().SetColor(*axis_colors[i])
                axis_model.GetDisplayNode().SetLineWidth(3.0)
                axis_model.GetDisplayNode().SetVisibility(0)  # Initially hidden