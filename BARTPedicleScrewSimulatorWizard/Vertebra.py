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
            
    def _detect_spinal_canal_improved(self, target_level=None):
        """
        Detect the anterior and posterior boundaries of the spinal canal
        using floodfill-inspired approach similar to the MATLAB implementation.
        Now supports level-specific detection.
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
        
        Returns:
            tuple: (canal_min_y, canal_max_y) in image coordinates
        """
        try:
            # If no centroid, return default values
            if not hasattr(self, 'centroid') or self.centroid is None:
                canal_min_y = 0
                canal_max_y = self.mask_array.shape[1] // 2
                self.logger.warning("No centroid available for spinal canal detection")
                return canal_min_y, canal_max_y
            
            # Create a mask that isolates only the target vertebra level
            level_mask = None
            if target_level:
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
                
                # Create level-specific mask
                if level_label is not None:
                    level_mask = (self.mask_array == level_label)
                    self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(level_mask)} voxels")
                    
                    # If the mask is empty, fall back to using all segmentations
                    if np.count_nonzero(level_mask) == 0:
                        self.logger.warning(f"Empty mask for label {level_label}, using all segmentations")
                        level_mask = None
            
            # Use level-specific mask if available, otherwise use the full mask
            binary_mask = level_mask if level_mask is not None else (self.mask_array > 0)
            
            # Convert centroid to IJK coordinates
            centroid_ijk = self._ras_to_ijk(self.centroid)
            
            # Get mid-axial slice at centroid Z
            sliceZ = int(np.round(centroid_ijk[2]))
            sliceZ = max(0, min(sliceZ, binary_mask.shape[2] - 1))
            
            # Extract slice
            slice_mask = binary_mask[:, :, sliceZ].copy()
            
            # Create a debug volume for visualization
            if hasattr(slicer, 'mrmlScene'):
                # Save binary mask for debugging
                debug_node = self._numpy_to_volume(
                    slice_mask[:, :, np.newaxis],  # Convert 2D slice to 3D volume
                    self.volume_node
                )
                if debug_node:
                    debug_node.SetName("DebugVertebraSlice")
            
            # Find the center point of the vertebra in this slice
            # Calculate centroid of the vertebra in this slice
            if np.count_nonzero(slice_mask) > 0:
                # Find the centroid specifically of this slice
                x_indices, y_indices = np.where(slice_mask > 0)
                x_idx = int(np.mean(x_indices))
                y_idx = int(np.mean(y_indices))
            else:
                # Fall back to the projected centroid
                x_idx = int(np.round(centroid_ijk[0]))
                x_idx = max(0, min(x_idx, slice_mask.shape[0] - 1))
                y_idx = int(np.round(centroid_ijk[1]))
                y_idx = max(0, min(y_idx, slice_mask.shape[1] - 1))
            
            # Store the initial indices for debugging
            initial_x, initial_y = x_idx, y_idx
            
            # 1. Move backward until we exit the vertebra
            posterior_edge = y_idx
            while posterior_edge > 0 and slice_mask[x_idx, posterior_edge] > 0:
                posterior_edge -= 1
                
            self.logger.info(f"Found posterior edge at y={posterior_edge}")
            
            # 2. Move forward from posterior edge until we enter the vertebra again
            anterior_edge = posterior_edge
            while anterior_edge < slice_mask.shape[1] - 1 and slice_mask[x_idx, anterior_edge] == 0:
                anterior_edge += 1
                
            self.logger.info(f"Found anterior edge at y={anterior_edge}")
            
            # If we found a valid canal (gap between vertebra segments)
            if anterior_edge > posterior_edge:
                # Use the floodfill approach to identify the canal more accurately
                try:
                    from scipy.ndimage import label
                    
                    # Seed point in the middle of the suspected canal
                    seed_y = (posterior_edge + anterior_edge) // 2
                    
                    # Create a marker for the seed
                    seed_mask = np.zeros_like(slice_mask)
                    seed_mask[x_idx, seed_y] = 1
                    
                    # Create inverse mask (0=vertebra, 1=background and canal)
                    inv_mask = 1 - slice_mask
                    
                    # Label connected regions
                    labeled_regions, num_regions = label(inv_mask)
                    
                    # Find the region containing our seed
                    canal_region = labeled_regions[x_idx, seed_y]
                    
                    # Create canal mask
                    canal_mask = (labeled_regions == canal_region)
                    
                    # Find Y boundaries of the canal
                    canal_y_indices = np.where(canal_mask.sum(axis=0) > 0)[0]
                    
                    if len(canal_y_indices) > 0:
                        canal_min_y = np.min(canal_y_indices)
                        canal_max_y = np.max(canal_y_indices)
                        
                        # Create a debug volume for the canal mask
                        if hasattr(slicer, 'mrmlScene'):
                            debug_canal = self._numpy_to_volume(
                                canal_mask[:, :, np.newaxis],
                                self.volume_node
                            )
                            if debug_canal:
                                debug_canal.SetName("DebugCanalMask")
                        
                        self.logger.info(f"Found canal boundaries using floodfill: min_y={canal_min_y}, max_y={canal_max_y} for level {target_level}")
                        
                        # Add margin to make sure we fully capture the canal
                        canal_min_y = max(0, canal_min_y - 5)
                        canal_max_y = min(slice_mask.shape[1] - 1, canal_max_y + 5)
                        
                        return canal_min_y, canal_max_y
                except Exception as e:
                    self.logger.warning(f"Floodfill method failed: {str(e)}")
                    # Fall back to the simpler method
                    
            # Fallback approach based on edge detection
            # Add margin to make sure we fully capture the canal
            canal_min_y = max(0, posterior_edge - 5)
            canal_max_y = min(slice_mask.shape[1] - 1, anterior_edge + 5)
            
            self.logger.info(f"Found canal boundaries using edge detection: min_y={canal_min_y}, max_y={canal_max_y} for level {target_level}")
            
            return canal_min_y, canal_max_y
            
        except Exception as e:
            self.logger.error(f"Error in _detect_spinal_canal_improved: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Default values on error
            canal_min_y = 0
            canal_max_y = self.mask_array.shape[1] // 2
            return canal_min_y, canal_max_y
    
    def _cut_pedicle(self, volume_array, y_max, y_min, buffer_front=0, buffer_end=0):
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
        try:
            # Create a copy to avoid modifying the original
            result = volume_array.copy()
            
            # Log the input parameters for debugging
            self.logger.info(f"Cutting pedicle with y_min={y_min}, y_max={y_max}, buffer_front={buffer_front}, buffer_end={buffer_end}")
            self.logger.info(f"Volume array shape: {result.shape}")
            
            # Calculate boundaries
            y_min_idx = max(0, min(int(y_min + buffer_front + 1), result.shape[1] - 1))
            y_max_idx = max(0, min(int(y_max - buffer_end), result.shape[1] - 1))
            
            self.logger.info(self.logger.info(f"Cutting pedicle at indices: y_min_idx={y_min_idx}, y_max_idx={y_max_idx}"))
            
            # Zero out regions outside pedicle (anterior portion)
            if y_min_idx > 0:
                result[:, 0:y_min_idx, :] = 0
                
            # Zero out regions outside pedicle (posterior portion)
            if y_max_idx < result.shape[1] - 1:
                result[:, y_max_idx:, :] = 0
            
            # Check if we have a valid result
            non_zero_count = np.count_nonzero(result)
            self.logger.info(f"Pedicle non-zero voxel count: {non_zero_count}")
            
            if non_zero_count == 0:
                self.logger.warning("Pedicle cutting resulted in empty array, using original volume")
                # If the cutting resulted in an empty array, return a small region around the centroid
                centroid_ijk = self._ras_to_ijk(self.centroid)
                x, y, z = [int(round(c)) for c in centroid_ijk]
                
                # Create a small box around the centroid
                radius = 10  # 10 voxels in each direction
                x_min = max(0, x - radius)
                x_max = min(volume_array.shape[0], x + radius)
                y_min = max(0, y - radius)
                y_max = min(volume_array.shape[1], y + radius)
                z_min = max(0, z - radius)
                z_max = min(volume_array.shape[2], z + radius)
                
                # Zero out everything except the small box
                result = volume_array.copy()
                result[:x_min, :, :] = 0
                result[x_max:, :, :] = 0
                result[:, :y_min, :] = 0
                result[:, y_max:, :] = 0
                result[:, :, :z_min] = 0
                result[:, :, z_max:] = 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _cut_pedicle: {str(e)}")
            self.logger.error(traceback.format_exc())
            return volume_array  # Return original on error
    
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
        
    def _cut_pedicle_with_label(self, volume_array, mask_array, y_max, y_min, target_level, buffer_front=0, buffer_end=0):
        """
        Cut out the pedicle region from the volume, using segmentation label information
        to isolate the specific vertebra of interest. Uses PCA-based alignment and 
        Z-direction seed point selection for better handling of tilted vertebrae.
        
        Parameters:
            volume_array: 3D numpy array of the volume
            mask_array: 3D numpy array of the segmentation mask
            y_max: Maximum Y coordinate of spinal canal (will be recalculated for specific level)
            y_min: Minimum Y coordinate of spinal canal (will be recalculated for specific level)
            target_level: Target level string (e.g., 'L4')
            buffer_front: Buffer before spinal canal
            buffer_end: Buffer after spinal canal
            
        Returns:
            numpy.ndarray: Cropped array containing only the pedicle region at the specified level
        """
        try:
            import time
            from scipy.spatial.transform import Rotation as R
            
            start_time = time.time()
            
            # Create a copy to avoid modifying the original
            result = volume_array.copy()
            
            # Log the input parameters for debugging
            self.logger.info(f"Cutting pedicle at {target_level} with original y_min={y_min}, y_max={y_max}")
            
            # Extract the vertebra label for the target level
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
            
            # Create a mask for the specific vertebra if we have a label
            vertebra_mask = np.ones_like(mask_array)
            if level_label is not None:
                vertebra_mask = (mask_array == level_label)
                self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(vertebra_mask)} voxels")
            
                # If the mask is empty, fall back to using any segmentation
                if np.count_nonzero(vertebra_mask) == 0:
                    self.logger.warning(f"Empty mask for label {level_label}, using any segmentation")
                    vertebra_mask = (mask_array > 0)
            else:
                # If we can't determine the label, use any segmentation
                vertebra_mask = (mask_array > 0)
            
            # Apply the vertebra mask to isolate the target vertebra
            masked_volume = result * vertebra_mask
            
            # Step 1: Use the pre-computed PCA results from initialization if available
            if hasattr(self, 'pcaVectors') and hasattr(self, 'best_aligned_pca_idx') and hasattr(self, 'vertebra_center_ras'):
                pca_start_time = time.time()
                
                # Get the best-aligned principal axis
                main_axis = self.pcaVectors[:, self.best_aligned_pca_idx]
                main_axis = main_axis / np.linalg.norm(main_axis)  # Ensure it's normalized
                
                # Create a rotation matrix to align the main axis with the Y axis (anterior-posterior)
                target_axis = np.array([0, 1, 0])  # Align with Y axis
                
                # Use the vertebra center point
                center_of_mass_ras = self.vertebra_center_ras
                
                # Convert to IJK coordinates for the transformation
                center_of_mass = self._ras_to_ijk(center_of_mass_ras)
                
                # Create rotation that aligns main_axis with the target_axis
                rotation = R.align_vectors([target_axis], [main_axis])[0]
                rotation_matrix = rotation.as_matrix()
                
                self.logger.info(f"Using pre-computed PCA axis {self.best_aligned_pca_idx} for alignment")
                self.logger.info(f"Alignment score: {getattr(self, 'pca_alignment_score', 'unknown')}")
                self.logger.info(f"Using vertebra center: {center_of_mass}")
            else:
                # Fallback: Calculate PCA in this function if not already available
                self.logger.warning("No pre-computed PCA available, calculating in _cut_pedicle_with_label")
                
                pca_start_time = time.time()
                
                # Get coordinates of all non-zero voxels in the vertebra mask
                nonzero_coords = np.where(vertebra_mask)
                if len(nonzero_coords[0]) < 10:  # Need enough points for meaningful PCA
                    self.logger.warning(f"Not enough voxels ({len(nonzero_coords[0])}) for PCA")
                    # Fall back to original canal detection
                    level_y_min, level_y_max = self._detect_spinal_canal_improved(target_level)
                    
                    # Apply original cutting method
                    y_min_idx = max(0, min(int(level_y_min + buffer_front + 1), masked_volume.shape[1] - 1))
                    y_max_idx = max(0, min(int(level_y_max - buffer_end), masked_volume.shape[1] - 1))
                    
                    pedicle_result = masked_volume.copy()
                    pedicle_result[:, 0:y_min_idx, :] = 0
                    pedicle_result[:, y_max_idx:, :] = 0
                    
                    self.logger.info(f"Fallback pedicle cutting took {time.time() - start_time:.2f} seconds")
                    return pedicle_result
                
                # Convert to (n_points, n_dims) array for PCA
                points = np.vstack([nonzero_coords[0], nonzero_coords[1], nonzero_coords[2]]).T
                
                # Perform PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(points)
                
                # Get principal components
                principal_axes = pca.components_
                
                # Use the select_best_alignment_vector function to find the best axis
                from .PCAUtils import select_best_alignment_vector
                target_axis = np.array([0, 1, 0])  # Y-axis (posterior-anterior direction)
                best_idx, main_axis, alignment_score = select_best_alignment_vector(principal_axes, target_axis)
                
                self.logger.info(f"Selected PCA axis {best_idx} with alignment score {alignment_score:.4f} to Y-axis")
                
                # Calculate the center of mass of the vertebra
                center_of_mass = np.mean(points, axis=0)
                
                # Create rotation that aligns main_axis with the target_axis
                rotation = R.align_vectors([target_axis], [main_axis])[0]
                rotation_matrix = rotation.as_matrix()
            
            self.logger.info(f"PCA/alignment lookup took {time.time() - pca_start_time:.2f} seconds")
            
            # Step 2: Transform the vertebra to aligned space
            transform_start_time = time.time()
            
            # Create aligned volume with same shape as original
            aligned_volume = np.zeros_like(masked_volume)
            aligned_mask = np.zeros_like(vertebra_mask)
            
            # For each voxel in the original vertebra
            original_indices = np.where(vertebra_mask)
            original_points = np.vstack([original_indices[0], original_indices[1], original_indices[2]]).T
            
            # Center the points
            centered_points = original_points - center_of_mass
            
            # Apply rotation to get aligned points
            aligned_points = (rotation.apply(centered_points) + center_of_mass)
            
            # Round to get voxel indices
            aligned_indices = np.round(aligned_points).astype(int)
            
            # Filter to points within bounds
            valid_mask = (
                (aligned_indices[:, 0] >= 0) & 
                (aligned_indices[:, 0] < masked_volume.shape[0]) &
                (aligned_indices[:, 1] >= 0) & 
                (aligned_indices[:, 1] < masked_volume.shape[1]) &
                (aligned_indices[:, 2] >= 0) & 
                (aligned_indices[:, 2] < masked_volume.shape[2])
            )
            
            valid_aligned_indices = aligned_indices[valid_mask]
            valid_original_indices = original_points[valid_mask].astype(int)
            
            # Map values from original to aligned space
            for i in range(len(valid_aligned_indices)):
                aligned_idx = tuple(valid_aligned_indices[i])
                orig_idx = tuple(valid_original_indices[i])
                aligned_volume[aligned_idx] = masked_volume[orig_idx]
                aligned_mask[aligned_idx] = 1
            
            # Create debug volumes for visualization
            if hasattr(slicer, 'mrmlScene'):
                debug_aligned = self._numpy_to_volume(
                    aligned_volume,
                    self.volume_node
                )
                if debug_aligned:
                    debug_aligned.SetName(f"DebugAlignedVertebra_{target_level}")
            
            self.logger.info(f"Transformation took {time.time() - transform_start_time:.2f} seconds")
            
            # Step 3: Detect spinal canal in aligned space using the improved Z-direction method
            canal_detection_start = time.time()
            
            # Use the dedicated function for Z-direction seed point selection
            aligned_y_min, aligned_y_max = self._find_spinal_canal_in_aligned_space(aligned_mask)
            
            self.logger.info(f"Aligned space canal boundaries: min_y={aligned_y_min}, max_y={aligned_y_max}")
            self.logger.info(f"Canal detection took {time.time() - canal_detection_start:.2f} seconds")
            
            # Step 4: Cut pedicle in aligned space
            cutting_start = time.time()
            
            # Apply the buffer in aligned space
            aligned_y_min_idx = max(0, min(int(aligned_y_min + buffer_front + 1), aligned_volume.shape[1] - 1))
            aligned_y_max_idx = max(0, min(int(aligned_y_max - buffer_end), aligned_volume.shape[1] - 1))
            
            # Create a copy to work on
            aligned_pedicle = aligned_volume.copy()
            
            # Zero out regions outside pedicle (anterior and posterior)
            aligned_pedicle[:, 0:aligned_y_min_idx, :] = 0
            aligned_pedicle[:, aligned_y_max_idx:, :] = 0
            
            # Create debug pedicle visualization in aligned space
            if hasattr(slicer, 'mrmlScene'):
                debug_aligned_pedicle = self._numpy_to_volume(
                    aligned_pedicle,
                    self.volume_node
                )
                if debug_aligned_pedicle:
                    debug_aligned_pedicle.SetName(f"DebugAlignedPedicle_{target_level}")
            
            self.logger.info(f"Cutting took {time.time() - cutting_start:.2f} seconds")
            
            # Step 5: Transform pedicle back to original space
            transform_back_start = time.time()
            
            # Create output array in original space
            pedicle_result = np.zeros_like(masked_volume)
            
            # Get indices of the pedicle in aligned space
            aligned_pedicle_indices = np.where(aligned_pedicle > 0)
            if len(aligned_pedicle_indices[0]) > 0:
                aligned_pedicle_points = np.vstack([
                    aligned_pedicle_indices[0], 
                    aligned_pedicle_indices[1], 
                    aligned_pedicle_indices[2]
                ]).T
                
                # Center the points
                centered_pedicle_points = aligned_pedicle_points - center_of_mass
                
                # Apply inverse rotation
                inverse_rotation = rotation.inv()
                original_pedicle_points = (inverse_rotation.apply(centered_pedicle_points) + center_of_mass)
                
                # Round to get voxel indices
                original_pedicle_indices = np.round(original_pedicle_points).astype(int)
                
                # Filter valid indices
                valid_mask = (
                    (original_pedicle_indices[:, 0] >= 0) & 
                    (original_pedicle_indices[:, 0] < pedicle_result.shape[0]) &
                    (original_pedicle_indices[:, 1] >= 0) & 
                    (original_pedicle_indices[:, 1] < pedicle_result.shape[1]) &
                    (original_pedicle_indices[:, 2] >= 0) & 
                    (original_pedicle_indices[:, 2] < pedicle_result.shape[2])
                )
                
                valid_original_indices = original_pedicle_indices[valid_mask]
                valid_aligned_indices = aligned_pedicle_points[valid_mask].astype(int)
                
                # Map values from aligned space back to original space
                for i in range(len(valid_original_indices)):
                    orig_idx = tuple(valid_original_indices[i])
                    aligned_idx = tuple(valid_aligned_indices[i])
                    pedicle_result[orig_idx] = aligned_pedicle[aligned_idx]
                
                self.logger.info(f"Transform back took {time.time() - transform_back_start:.2f} seconds")
            else:
                self.logger.warning("No pedicle voxels found in aligned space")
                # Fall back to using original (unaligned) approach
                level_y_min, level_y_max = self._detect_spinal_canal_improved(target_level)
                
                # Apply original cutting method
                y_min_idx = max(0, min(int(level_y_min + buffer_front + 1), masked_volume.shape[1] - 1))
                y_max_idx = max(0, min(int(level_y_max - buffer_end), masked_volume.shape[1] - 1))
                
                pedicle_result = masked_volume.copy()
                pedicle_result[:, 0:y_min_idx, :] = 0
                pedicle_result[:, y_max_idx:, :] = 0
            
            # Check if we have a valid result
            non_zero_count = np.count_nonzero(pedicle_result)
            self.logger.info(f"Pedicle non-zero voxel count: {non_zero_count}")
            
            if non_zero_count == 0:
                self.logger.warning("Pedicle cutting resulted in empty array, using fallback approach")
                
                # Fall back to a simple box around the vertebra centroid
                vertebra_indices = np.where(vertebra_mask)
                if len(vertebra_indices[0]) > 0:
                    v_centroid_ijk = np.array([
                        np.mean(vertebra_indices[0]),
                        np.mean(vertebra_indices[1]),
                        np.mean(vertebra_indices[2])
                    ]).astype(int)
                    
                    # Create a small box around the vertebra centroid
                    radius = 15  # 15 voxels in each direction
                    x_min = max(0, v_centroid_ijk[0] - radius)
                    x_max = min(volume_array.shape[0], v_centroid_ijk[0] + radius)
                    y_min = max(0, v_centroid_ijk[1] - radius)
                    y_max = min(volume_array.shape[1], v_centroid_ijk[1] + radius)
                    z_min = max(0, v_centroid_ijk[2] - radius)
                    z_max = min(volume_array.shape[2], v_centroid_ijk[2] + radius)
                    
                    # Zero out everything except the small box
                    pedicle_result = volume_array.copy() * vertebra_mask
                    pedicle_result[:x_min, :, :] = 0
                    pedicle_result[x_max:, :, :] = 0
                    pedicle_result[:, :y_min, :] = 0
                    pedicle_result[:, y_max:, :] = 0
                    pedicle_result[:, :, :z_min] = 0
                    pedicle_result[:, :, z_max:] = 0
                else:
                    self.logger.warning("Empty vertebra mask, using original volume")
                    # If still empty, return the original volume
                    pedicle_result = volume_array.copy()
            
            self.logger.info(f"Pedicle cutting with alignment took {time.time() - start_time:.2f} seconds")
            return pedicle_result
            
        except Exception as e:
            self.logger.error(f"Error in _cut_pedicle_with_label: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return volume_array  # Return original on error
        
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
    def _detect_spinal_canal_with_pca_alignment(self, target_level=None):
        """
        Detect the anterior and posterior boundaries of the spinal canal
        using PCA-based alignment of the vertebra to handle tilted vertebrae.
        Uses the best-aligned PCA axis computed during initialization.
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
        
        Returns:
            tuple: (canal_min_y, canal_max_y) in image coordinates
        """
        try:
            import time
            start_time = time.time()
            self.logger.info(f"Starting PCA-based spinal canal detection for level {target_level}")
            
            # If no centroid, return default values
            if not hasattr(self, 'centroid') or self.centroid is None:
                canal_min_y = 0
                canal_max_y = self.mask_array.shape[1] // 2
                self.logger.warning("No centroid available for spinal canal detection")
                return canal_min_y, canal_max_y
            
            # Create a mask that isolates only the target vertebra level
            level_mask = None
            if target_level:
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
                
                # Create level-specific mask
                if level_label is not None:
                    level_mask = (self.mask_array == level_label)
                    self.logger.info(f"Created mask for label {level_label} with {np.count_nonzero(level_mask)} voxels")
                    
                    # If the mask is empty, fall back to using all segmentations
                    if np.count_nonzero(level_mask) == 0:
                        self.logger.warning(f"Empty mask for label {level_label}, using all segmentations")
                        level_mask = None
            
            # Use level-specific mask if available, otherwise use the full mask
            binary_mask = level_mask if level_mask is not None else (self.mask_array > 0)
            
            # Report the time taken for mask creation
            self.logger.info(f"Mask creation took {time.time() - start_time:.2f} seconds")
            
            # Step 1: Use the pre-computed PCA results from initialization if available
            pca_start_time = time.time()
            
            if hasattr(self, 'pcaVectors') and hasattr(self, 'best_aligned_pca_idx') and hasattr(self, 'vertebra_center_ras'):
                # Get the best-aligned principal axis
                main_axis = self.pcaVectors[:, self.best_aligned_pca_idx]
                main_axis = main_axis / np.linalg.norm(main_axis)  # Ensure it's normalized
                
                # Create a rotation matrix to align the main axis with the Y axis (anterior-posterior)
                from scipy.spatial.transform import Rotation as R
                
                target_axis = np.array([0, 1, 0])  # Y-axis (anterior-posterior direction)
                
                # Use the vertebra center point
                center_of_mass_ras = self.vertebra_center_ras
                
                # Convert to IJK coordinates for the transformation
                center_of_mass = self._ras_to_ijk(center_of_mass_ras)
                
                # Create rotation that aligns main_axis with the target_axis
                rotation = R.align_vectors([target_axis], [main_axis])[0]
                rotation_matrix = rotation.as_matrix()
                
                self.logger.info(f"Using pre-computed PCA axis {self.best_aligned_pca_idx} for alignment")
                self.logger.info(f"Alignment score: {getattr(self, 'pca_alignment_score', 'unknown')}")
            else:
                # Perform PCA on the binary mask if pre-computed is not available
                self.logger.warning("No pre-computed PCA available, calculating in _detect_spinal_canal_with_pca_alignment")
                
                # Get coordinates of all non-zero voxels in the binary mask
                nonzero_coords = np.where(binary_mask)
                if len(nonzero_coords[0]) < 10:  # Need enough points for meaningful PCA
                    self.logger.warning(f"Not enough voxels ({len(nonzero_coords[0])}) for PCA")
                    # Fall back to original method
                    return self._detect_spinal_canal_improved(target_level)
                
                # Convert to (n_points, n_dims) array for PCA
                points = np.vstack([nonzero_coords[0], nonzero_coords[1], nonzero_coords[2]]).T
                
                # Perform PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(points)
                
                # Get principal components
                principal_axes = pca.components_
                
                # Find the axis that best aligns with the Y axis (anterior-posterior)
                from .PCAUtils import select_best_alignment_vector
                target_axis = np.array([0, 1, 0])  # Y-axis (anterior-posterior direction)
                best_idx, main_axis, alignment_score = select_best_alignment_vector(principal_axes, target_axis)
                
                self.logger.info(f"Selected PCA axis {best_idx} with alignment score {alignment_score:.4f} to Y-axis")
                
                # Calculate the center of mass
                center_of_mass = np.mean(points, axis=0)
                
                # Create rotation that aligns main_axis with the target_axis
                from scipy.spatial.transform import Rotation as R
                rotation = R.align_vectors([target_axis], [main_axis])[0]
                rotation_matrix = rotation.as_matrix()
            
            # Report the time taken for PCA
            self.logger.info(f"PCA took {time.time() - pca_start_time:.2f} seconds")
            
            # Now we need to apply this rotation to our binary mask
            transform_start_time = time.time()
            
            # Get the original image dimensions
            original_shape = binary_mask.shape
            
            # Create a coordinate grid in the original space
            x, y, z = np.meshgrid(
                np.arange(original_shape[0]),
                np.arange(original_shape[1]),
                np.arange(original_shape[2]),
                indexing='ij'
            )
            
            # Flatten the coordinate grid
            coords = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
            
            # Center coordinates
            centered_coords = coords - center_of_mass
            
            # Apply rotation to get new coordinates
            rotated_coords = np.dot(centered_coords, rotation_matrix.T) + center_of_mass
            
            # Round to nearest integer and convert to indices
            indices = np.round(rotated_coords).astype(int)
            
            # Filter out indices that are outside the original image bounds
            valid_indices = (
                (indices[:, 0] >= 0) & (indices[:, 0] < original_shape[0]) &
                (indices[:, 1] >= 0) & (indices[:, 1] < original_shape[1]) &
                (indices[:, 2] >= 0) & (indices[:, 2] < original_shape[2])
            )
            
            # Create the aligned binary mask
            aligned_mask = np.zeros_like(binary_mask)
            
            # Map values from original mask to aligned mask
            for i in range(len(indices)):
                if valid_indices[i]:
                    # Get the original coordinate
                    orig_x, orig_y, orig_z = coords[i].astype(int)
                    
                    # Get the rotated coordinate
                    rot_x, rot_y, rot_z = indices[i]
                    
                    # Copy value from original to rotated position
                    if binary_mask[orig_x, orig_y, orig_z]:
                        aligned_mask[rot_x, rot_y, rot_z] = 1
            
            # Report the time taken for transformation
            self.logger.info(f"Transformation took {time.time() - transform_start_time:.2f} seconds")
            
            # Now we can detect the spinal canal in the aligned mask
            canal_start_time = time.time()
            
            # Create a debug volume for visualization
            if hasattr(slicer, 'mrmlScene'):
                debug_aligned = self._numpy_to_volume(
                    aligned_mask,
                    self.volume_node
                )
                if debug_aligned:
                    debug_aligned.SetName("DebugAlignedVertebraMask")
            
            # Calculate the center of mass in the aligned space
            aligned_points = np.where(aligned_mask)
            aligned_center = np.mean(np.vstack([aligned_points[0], aligned_points[1], aligned_points[2]]).T, axis=0)
            
            # Get a mid-axial slice at the center of mass Z
            sliceZ = int(np.round(aligned_center[2]))
            sliceZ = max(0, min(sliceZ, aligned_mask.shape[2] - 1))
            
            # Extract slice
            slice_mask = aligned_mask[:, :, sliceZ].copy()
            
            # Find the center point of the vertebra in this slice
            if np.count_nonzero(slice_mask) > 0:
                # Find the centroid of this slice
                x_indices, y_indices = np.where(slice_mask > 0)
                x_idx = int(np.mean(x_indices))
                y_idx = int(np.mean(y_indices))
            else:
                # Fall back to the center of the image
                x_idx = slice_mask.shape[0] // 2
                y_idx = slice_mask.shape[1] // 2
                self.logger.warning("No voxels in slice for centroid calculation, using center of image")
            
            # Store the initial indices for debugging
            initial_x, initial_y = x_idx, y_idx
            
            # 1. Move backward until we exit the vertebra
            posterior_edge = y_idx
            while posterior_edge > 0 and slice_mask[x_idx, posterior_edge] > 0:
                posterior_edge -= 1
                
            self.logger.info(f"Found posterior edge at y={posterior_edge}")
            
            # 2. Move forward from posterior edge until we enter the vertebra again
            anterior_edge = posterior_edge
            while anterior_edge < slice_mask.shape[1] - 1 and slice_mask[x_idx, anterior_edge] == 0:
                anterior_edge += 1
                
            self.logger.info(f"Found anterior edge at y={anterior_edge}")
            
            # If we found a valid canal (gap between vertebra segments)
            if anterior_edge > posterior_edge:
                # Use the floodfill approach to identify the canal more accurately
                try:
                    from scipy.ndimage import label
                    
                    # Seed point in the middle of the suspected canal
                    seed_y = (posterior_edge + anterior_edge) // 2
                    
                    # Create a marker for the seed
                    seed_mask = np.zeros_like(slice_mask)
                    seed_mask[x_idx, seed_y] = 1
                    
                    # Create inverse mask (0=vertebra, 1=background and canal)
                    inv_mask = 1 - slice_mask
                    
                    # Label connected regions
                    labeled_regions, num_regions = label(inv_mask)
                    
                    # Find the region containing our seed
                    canal_region = labeled_regions[x_idx, seed_y]
                    
                    # Create canal mask
                    canal_mask = (labeled_regions == canal_region)
                    
                    # Find Y boundaries of the canal
                    canal_y_indices = np.where(canal_mask.sum(axis=0) > 0)[0]
                    
                    if len(canal_y_indices) > 0:
                        aligned_canal_min_y = np.min(canal_y_indices)
                        aligned_canal_max_y = np.max(canal_y_indices)
                        
                        # Create a debug volume for the canal mask
                        if hasattr(slicer, 'mrmlScene'):
                            canal_3d = np.zeros_like(aligned_mask)
                            canal_3d[:, :, sliceZ] = canal_mask
                            debug_canal = self._numpy_to_volume(
                                canal_3d,
                                self.volume_node
                            )
                            if debug_canal:
                                debug_canal.SetName("DebugCanalMask")
                        
                        self.logger.info(f"Found canal boundaries in aligned space: min_y={aligned_canal_min_y}, max_y={aligned_canal_max_y}")
                        
                        # Now we need to transform these boundaries back to the original space
                        # This is more complex since we need to map 2D slice coordinates
                        
                        # Use the inverse transformation to map points back
                        inverse_rotation = rotation_matrix.T  # Transpose = inverse for rotation matrices
                        
                        # Create a mask in the aligned space that only contains the canal
                        canal_line_mask = np.zeros_like(slice_mask)
                        canal_line_mask[x_idx, aligned_canal_min_y:aligned_canal_max_y + 1] = 1
                        
                        # Transform this mask back to the original space
                        # For simplicity, we'll use the same transformation approach but with inverse rotation
                        
                        # Get coordinates of non-zero voxels in the canal line mask
                        canal_coords = np.where(canal_line_mask)
                        
                        # Combine with Z coordinate to get 3D points
                        canal_points = np.vstack([canal_coords[0], canal_coords[1], np.full_like(canal_coords[0], sliceZ)]).T
                        
                        # Center coordinates
                        centered_canal_points = canal_points - center_of_mass
                        
                        # Apply inverse rotation
                        original_canal_points = np.dot(centered_canal_points, inverse_rotation.T) + center_of_mass
                        
                        # Project back to original Y coordinates
                        original_y_coords = original_canal_points[:, 1]
                        
                        # Find the range in the original space
                        if len(original_y_coords) > 0:
                            canal_min_y = int(np.min(original_y_coords))
                            canal_max_y = int(np.max(original_y_coords))
                            
                            # Add margin to make sure we fully capture the canal
                            canal_min_y = max(0, canal_min_y - 5)
                            canal_max_y = min(binary_mask.shape[1] - 1, canal_max_y + 5)
                            
                            self.logger.info(f"Transformed canal boundaries to original space: min_y={canal_min_y}, max_y={canal_max_y}")
                            
                            # Report the time taken for canal detection
                            self.logger.info(f"Canal detection took {time.time() - canal_start_time:.2f} seconds")
                            
                            return canal_min_y, canal_max_y
                        else:
                            self.logger.warning("No canal points found after inverse transformation")
                except Exception as e:
                    self.logger.warning(f"Floodfill method failed: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # Fallback approach: map aligned boundaries directly back to original space
            # We'll use the inverse of the transformation we performed
            
            # Calculate an approximate mapping from aligned to original
            # For simplicity, we'll just map the min and max Y values
            try:
                # Create test points at the aligned boundaries
                test_min_point = np.array([x_idx, posterior_edge, sliceZ])
                test_max_point = np.array([x_idx, anterior_edge, sliceZ])
                
                # Center these points
                centered_min = test_min_point - center_of_mass
                centered_max = test_max_point - center_of_mass
                
                # Apply inverse rotation
                original_min = np.dot(centered_min, inverse_rotation.T) + center_of_mass
                original_max = np.dot(centered_max, inverse_rotation.T) + center_of_mass
                
                # Extract Y coordinates
                canal_min_y = int(original_min[1])
                canal_max_y = int(original_max[1])
                
                # Add margin to make sure we fully capture the canal
                canal_min_y = max(0, canal_min_y - 5)
                canal_max_y = min(binary_mask.shape[1] - 1, canal_max_y + 5)
                
                self.logger.info(f"Fallback transformed canal boundaries: min_y={canal_min_y}, max_y={canal_max_y}")
                
                return canal_min_y, canal_max_y
                
            except Exception as e:
                self.logger.error(f"Error in canal boundary transformation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # If all else fails, use a simple heuristic
            canal_min_y = original_shape[1] // 3
            canal_max_y = 2 * original_shape[1] // 3
            
            self.logger.warning(f"Using default canal boundaries: min_y={canal_min_y}, max_y={canal_max_y}")
            
            return canal_min_y, canal_max_y
            
        except Exception as e:
            self.logger.error(f"Error in _detect_spinal_canal_with_pca_alignment: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Default values on error
            canal_min_y = 0
            canal_max_y = self.mask_array.shape[1] // 2
            return canal_min_y, canal_max_y
    
    def _find_spinal_canal_in_aligned_space(self, aligned_mask):
        """
        Find the spinal canal in the aligned vertebra space.
        Uses multiple Z-levels (superior-inferior) to find the best spinal canal seed point.
        
        Parameters:
            aligned_mask: Binary mask of the aligned vertebra
        
        Returns:
            tuple: (y_min, y_max) boundaries of the canal in aligned space
        """
        try:
            from scipy.ndimage import label, binary_erosion
            import time
            start_time = time.time()
            
            # Find vertebra extents in Z direction
            aligned_points = np.where(aligned_mask)
            if len(aligned_points[0]) == 0:
                self.logger.warning("No points in aligned mask")
                return aligned_mask.shape[1] // 3, 2 * aligned_mask.shape[1] // 3
            
            # Get Z range of the vertebra
            z_min = np.min(aligned_points[2])
            z_max = np.max(aligned_points[2])
            z_range = z_max - z_min + 1
            
            self.logger.info(f"Vertebra Z range: {z_min} to {z_max}")
            
            # Sample multiple slices through the Z range
            num_slices = min(10, z_range)  # Try up to 10 slices or as many as we have
            
            # We'll collect canal candidates across all slices
            all_canal_regions = []  # List of tuples: (score, z_level, y_min, y_max)
            
            # Try slices throughout the Z range
            for i in range(num_slices):
                # Calculate slice position - distribute evenly across the Z range
                if num_slices > 1:
                    z_level = int(z_min + (z_range * i) / (num_slices - 1))
                else:
                    z_level = int((z_min + z_max) / 2)
                
                # Ensure we stay within bounds
                z_level = max(z_min, min(z_level, z_max))
                
                # Extract slice at this Z level
                slice_mask = aligned_mask[:, :, z_level].copy()
                
                # Skip if empty slice
                if np.sum(slice_mask) == 0:
                    continue
                    
                # Find center of the vertebra in this slice
                slice_points = np.where(slice_mask)
                center_x = int(np.mean(slice_points[0]))
                center_y = int(np.mean(slice_points[1]))
                
                # Create inverse mask (0 for vertebra, 1 for background)
                inv_slice_mask = 1 - slice_mask
                
                # Find connected components in the inverse mask
                labeled_regions, num_regions = label(inv_slice_mask)
                
                # Skip if no regions found
                if num_regions == 0:
                    continue
                    
                # Find posterior edge
                posterior_edge = center_y
                while posterior_edge > 0 and slice_mask[center_x, posterior_edge] > 0:
                    posterior_edge -= 1
                    
                # Try to find a good canal region in this slice
                canal_region = None
                best_score = float('inf')
                region_y_min = 0
                region_y_max = 0
                
                # Check all regions
                for region_id in range(1, num_regions + 1):
                    region_mask = (labeled_regions == region_id)
                    region_size = np.sum(region_mask)
                    
                    # Skip regions that are too small or too large
                    if region_size < 10 or region_size > 1000:
                        continue
                        
                    # Get region extents
                    region_points = np.where(region_mask)
                    region_center_x = int(np.mean(region_points[0]))
                    region_center_y = int(np.mean(region_points[1]))
                    region_y_indices = region_points[1]
                    local_y_min = np.min(region_y_indices)
                    local_y_max = np.max(region_y_indices)
                    
                    # Calculate score based on:
                    # 1. X-distance from vertebra center (should be small)
                    # 2. Y-position relative to posterior edge (should be near posterior edge)
                    # 3. Size (should be reasonably small)
                    
                    x_distance = abs(region_center_x - center_x) / slice_mask.shape[0]
                    y_dist_from_edge = abs(region_center_y - posterior_edge) / slice_mask.shape[1]
                    size_factor = region_size / 500  # Normalize size (500 pixels is fairly large)
                    
                    # Lower score is better
                    score = x_distance * 3 + y_dist_from_edge * 2 + size_factor
                    
                    # Add a penalty if region is anterior to the center (likely not the canal)
                    if region_center_y > center_y:
                        score += 5
                    
                    if score < best_score:
                        best_score = score
                        canal_region = region_id
                        region_y_min = local_y_min
                        region_y_max = local_y_max
                
                # If we found a good canal region in this slice
                if canal_region is not None:
                    self.logger.info(f"Z-level {z_level}: Found canal candidate with score {best_score:.2f}, y_min={region_y_min}, y_max={region_y_max}")
                    all_canal_regions.append((best_score, z_level, region_y_min, region_y_max))
                    
                    # Create a debug visualization for this canal
                    if hasattr(slicer, 'mrmlScene'):
                        debug_canal = np.zeros_like(aligned_mask)
                        debug_canal[:, :, z_level] = (labeled_regions == canal_region)
                        debug_canal_node = self._numpy_to_volume(
                            debug_canal,
                            self.volume_node
                        )
                        if debug_canal_node:
                            debug_canal_node.SetName(f"DebugCanal_Z{z_level}")
            
            # If we found at least one good canal region
            if all_canal_regions:
                # Sort by score (lowest is best)
                all_canal_regions.sort(key=lambda x: x[0])
                
                # Choose the best one
                best_region = all_canal_regions[0]
                best_score, best_z, best_y_min, best_y_max = best_region
                
                self.logger.info(f"Best canal found at Z={best_z} with score {best_score:.2f}")
                
                # Add margins
                y_min = max(0, best_y_min - 5)
                y_max = min(aligned_mask.shape[1] - 1, best_y_max + 5)
                
                return y_min, y_max
            
            self.logger.warning("No good canal regions found across Z slices, using fallback method")
            
            # Fallback: use the middle slice
            middle_z = int((z_min + z_max) / 2)
            middle_slice = aligned_mask[:, :, middle_z].copy()
            
            # Find center of vertebra
            middle_points = np.where(middle_slice)
            if len(middle_points[0]) > 0:
                middle_x = int(np.mean(middle_points[0]))
                middle_y = int(np.mean(middle_points[1]))
                
                # Find posterior edge
                posterior_edge = middle_y
                while posterior_edge > 0 and middle_slice[middle_x, posterior_edge] > 0:
                    posterior_edge -= 1
                    
                # Find anterior edge
                anterior_edge = posterior_edge
                while anterior_edge < middle_slice.shape[1] - 1 and middle_slice[middle_x, anterior_edge] == 0:
                    anterior_edge += 1
                    
                self.logger.info(f"Fallback edges: posterior={posterior_edge}, anterior={anterior_edge}")
                
                # Add margins
                y_min = max(0, posterior_edge - 5)
                y_max = min(aligned_mask.shape[1] - 1, anterior_edge + 5)
                
                return y_min, y_max
                
            # If all else fails, use simple division
            y_min = aligned_mask.shape[1] // 3
            y_max = 2 * aligned_mask.shape[1] // 3
            
            self.logger.info(f"Using default canal boundaries: y_min={y_min}, y_max={y_max}")
            self.logger.info(f"Canal detection took {time.time() - start_time:.2f} seconds")
            
            return y_min, y_max
            
        except Exception as e:
            self.logger.error(f"Error finding spinal canal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Default to simple division
            return aligned_mask.shape[1] // 3, 2 * aligned_mask.shape[1] // 3
        
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
        Find the smallest cross-section of the pedicle by moving through the spinal canal.
        Modified to search primarily in the Y direction to find more accurate canal edges.
        
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
        
        # Convert vtkPolyData to numpy array
        volume_points = self._points_to_numpy(aligned_volume_cloud.GetPoints())
        
        if volume_points.shape[0] == 0:
            self.logger.warning("Empty point cloud, cannot find smallest cross-section")
            return 0, None, float('inf')
        
        # Transform insertion point to aligned space to determine side
        insertion_point_aligned = self.transform_to_aligned_space(
            self.insertion_point, center_of_mass, rotation_matrix)
        
        # Determine side of interest (left or right based on insertion point)
        is_left_side = insertion_point_aligned[0] < aligned_center[0]
        side_text = "left" if is_left_side else "right"
        self.logger.info(f"Working on {side_text} side based on insertion point at {insertion_point_aligned}")
        
        # Filter points based on side
        side_filter = volume_points[:, 0] < aligned_center[0] if is_left_side else volume_points[:, 0] > aligned_center[0]
        side_points = volume_points[side_filter]
        
        if len(side_points) == 0:
            self.logger.warning(f"No points found on the {side_text} side")
            return 0, None, float('inf')
        
        # Build KD-tree for efficient nearest neighbor searches
        tree = cKDTree(side_points)
        
        # Parameters for edge detection - these might need tuning
        sphere_radius = 5.0      # Search radius in mm
        point_threshold = 5      # Minimum points to consider as "edge"
        step_size = 1.0          # Step size for movement in mm
        search_range = 40.0      # Maximum distance to search for edges
        
        # Calculate the X position for the side of interest
        # Make sure we're far enough from the centerline to be in the pedicle region
        side_offset = 15.0  # mm - this is the distance from midline to ensure we're in the pedicle region
        test_x = aligned_center[0] - side_offset if is_left_side else aligned_center[0] + side_offset
        
        # Get range of Y values to search for the canal
        y_range = np.percentile(side_points[:, 1], [20, 80])  # Use 20th-80th percentile as search range
        y_min, y_max = y_range[0] - 10, y_range[1] + 10  # Add margin
        
        # Create debug markers for the search process
        if hasattr(slicer, 'mrmlScene'):
            debug_fiducials = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PedicleVoidPoints")
            debug_fiducials.CreateDefaultDisplayNodes()
            debug_fiducials.GetDisplayNode().SetSelectedColor(0.0, 1.0, 0.0)  # Green
        
        # Find a void point by searching systematically in the Y direction
        void_y = None
        test_z = aligned_center[2]  # Use Z coordinate of the centroid
        
        # Create a grid of Y positions to search
        y_steps = np.arange(y_min, y_max, 5.0)  # 5mm steps for initial search
        
        # Search for a void point - only vary Y, keep X fixed
        for y_pos in y_steps:
            test_point = np.array([test_x, y_pos, test_z])
            indices = tree.query_ball_point(test_point, sphere_radius)
            
            self.logger.debug(f"Testing y={y_pos}, found {len(indices)} points in sphere")
            
            # If we have few enough points, we might be in the void
            if len(indices) < point_threshold:
                void_y = y_pos
                self.logger.info(f"Found potential void at y={void_y}")
                
                # Add debug marker for potential void point
                if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                    idx = debug_fiducials.AddFiducial(test_x, void_y, test_z)
                    debug_fiducials.SetNthFiducialLabel(idx, "Potential Void")
                
                # Verify this is truly a void by checking nearby points
                # Ensure we're truly in a void, not just in a small gap
                is_true_void = True
                for check_offset in [-5, 0, 5]:
                    check_point = np.array([test_x, void_y + check_offset, test_z])
                    check_indices = tree.query_ball_point(check_point, sphere_radius)
                    if len(check_indices) >= point_threshold:
                        is_true_void = False
                        break
                
                if is_true_void:
                    self.logger.info(f"Confirmed void at y={void_y}")
                    # Add debug marker for confirmed void
                    if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                        idx = debug_fiducials.AddFiducial(test_x, void_y, test_z)
                        debug_fiducials.SetNthFiducialLabel(idx, "Confirmed Void")
                    break
                else:
                    void_y = None  # Reset if this isn't a true void
        
        # If we couldn't find a void point with the initial search, try a more thorough search
        if void_y is None:
            self.logger.info("Initial void search failed, trying more thorough search...")
            
            # Try different X offsets, but only if necessary
            # We want to stay on the correct side of the vertebra
            x_range = [test_x - 5, test_x, test_x + 5] if is_left_side else [test_x + 5, test_x, test_x - 5]
            
            # Constrain X range to stay on the correct side
            if is_left_side:
                x_range = [x for x in x_range if x < aligned_center[0]]
            else:
                x_range = [x for x in x_range if x > aligned_center[0]]
            
            # Try a finer Y grid
            y_steps = np.arange(y_min, y_max, 2.0)  # 2mm steps for more thorough search
            
            for x_pos in x_range:
                for y_pos in y_steps:
                    test_point = np.array([x_pos, y_pos, test_z])
                    indices = tree.query_ball_point(test_point, sphere_radius)
                    
                    if len(indices) < point_threshold:
                        # Verify this is truly a void by checking nearby points
                        is_true_void = True
                        for check_offset in [-3, 0, 3]:
                            check_point = np.array([x_pos, y_pos + check_offset, test_z])
                            check_indices = tree.query_ball_point(check_point, sphere_radius)
                            if len(check_indices) >= point_threshold:
                                is_true_void = False
                                break
                        
                        if is_true_void:
                            void_y = y_pos
                            test_x = x_pos  # Update test_x to the working position
                            self.logger.info(f"Found void in thorough search at x={test_x}, y={void_y}")
                            
                            # Add debug marker
                            if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                                idx = debug_fiducials.AddFiducial(test_x, void_y, test_z)
                                debug_fiducials.SetNthFiducialLabel(idx, "Thorough Search Void")
                            break
                
                if void_y is not None:
                    break
        
        # If we still couldn't find a void, try a statistical approach
        if void_y is None:
            self.logger.warning("Could not find void point, using statistical approach")
            
            # Use statistics to estimate where the canal should be
            # The spinal canal is typically around the middle of the y-range
            y_percentiles = np.percentile(side_points[:, 1], [25, 40, 50, 60, 75])
            
            # Try several potential Y positions based on percentiles
            for y_pos in y_percentiles:
                test_point = np.array([test_x, y_pos, test_z])
                indices = tree.query_ball_point(test_point, sphere_radius * 1.5)  # Use larger radius
                
                if len(indices) < point_threshold * 2:  # More lenient threshold
                    void_y = y_pos
                    self.logger.info(f"Using statistical void at y={void_y}")
                    
                    # Add debug marker
                    if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                        idx = debug_fiducials.AddFiducial(test_x, void_y, test_z)
                        debug_fiducials.SetNthFiducialLabel(idx, "Statistical Void")
                    break
        
        # If we still couldn't find a void, use vertebra center Y as fallback
        if void_y is None:
            void_y = aligned_center[1]  # Use centroid Y
            self.logger.warning(f"Using centroid Y={void_y} as fallback")
            
            # Add debug marker
            if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                idx = debug_fiducials.AddFiducial(test_x, void_y, test_z)
                debug_fiducials.SetNthFiducialLabel(idx, "Fallback Void")
        
        # Now search for the posterior edge (decreasing Y)
        posterior_edge = void_y
        posterior_found = False
        
        current_point = np.array([test_x, void_y, test_z])
        
        # Add marker for starting void point
        if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
            idx = debug_fiducials.AddFiducial(current_point[0], current_point[1], current_point[2])
            debug_fiducials.SetNthFiducialLabel(idx, "Search Start")
        
        self.logger.info(f"Starting posterior search from y={posterior_edge}")
        
        # Set search bounds
        min_y = max(y_min, void_y - search_range)
        
        # Search posteriorly (decreasing Y) - use smaller step size for accuracy
        for i in range(int(search_range / (step_size/2))):
            current_point[1] -= step_size/2
            
            # Check if we're beyond bounds
            if current_point[1] < min_y:
                self.logger.warning(f"Reached minimum y ({min_y}) without finding posterior edge")
                break
            
            # Check if we hit the vertebra
            indices = tree.query_ball_point(current_point, sphere_radius)
            num_points = len(indices)
            
            # Log every few steps to reduce noise
            if i % 5 == 0:
                self.logger.debug(f"Posterior search at y={current_point[1]}, found {num_points} points")
            
            if num_points >= point_threshold:
                # Verify with additional checks to avoid false edges
                # Check a few nearby points to confirm we've truly hit an edge
                edge_confirmed = True
                for check_offset in [-1, 0, 1]:
                    check_x = current_point[0] + check_offset
                    check_point = np.array([check_x, current_point[1], current_point[2]])
                    check_indices = tree.query_ball_point(check_point, sphere_radius)
                    if len(check_indices) < point_threshold:
                        edge_confirmed = False
                        break
                
                if edge_confirmed:
                    posterior_edge = current_point[1]
                    posterior_found = True
                    self.logger.info(f"Hit posterior edge at y={posterior_edge}")
                    
                    # Add debug marker
                    if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                        idx = debug_fiducials.AddFiducial(current_point[0], current_point[1], current_point[2])
                        debug_fiducials.SetNthFiducialLabel(idx, "Posterior Edge")
                    
                    break
        
        # Now search for the anterior edge (increasing Y)
        anterior_edge = void_y
        anterior_found = False
        
        # Reset current point to start search from void
        current_point = np.array([test_x, void_y, test_z])
        self.logger.info(f"Starting anterior search from y={anterior_edge}")
        
        # Set search bounds
        max_y = min(y_max, void_y + search_range)
        
        # Search anteriorly (increasing Y) - use smaller step size for accuracy
        for i in range(int(search_range / (step_size/2))):
            current_point[1] += step_size/2
            
            # Check if we're beyond bounds
            if current_point[1] > max_y:
                self.logger.warning(f"Reached maximum y ({max_y}) without finding anterior edge")
                break
            
            # Check if we hit the vertebra
            indices = tree.query_ball_point(current_point, sphere_radius)
            num_points = len(indices)
            
            # Log every few steps to reduce noise
            if i % 5 == 0:
                self.logger.debug(f"Anterior search at y={current_point[1]}, found {num_points} points")
            
            if num_points >= point_threshold:
                # Verify with additional checks to avoid false edges
                edge_confirmed = True
                for check_offset in [-1, 0, 1]:
                    check_x = current_point[0] + check_offset
                    check_point = np.array([check_x, current_point[1], current_point[2]])
                    check_indices = tree.query_ball_point(check_point, sphere_radius)
                    if len(check_indices) < point_threshold:
                        edge_confirmed = False
                        break
                
                if edge_confirmed:
                    anterior_edge = current_point[1]
                    anterior_found = True
                    self.logger.info(f"Hit anterior edge at y={anterior_edge}")
                    
                    # Add debug marker
                    if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
                        idx = debug_fiducials.AddFiducial(current_point[0], current_point[1], current_point[2])
                        debug_fiducials.SetNthFiducialLabel(idx, "Anterior Edge")
                    
                    break
        
        # If we couldn't find the edges, use statistical approach
        if not posterior_found:
            posterior_edge = np.percentile(side_points[:, 1], 25)  # Use 25th percentile as posterior edge
            self.logger.warning(f"Using statistical posterior edge at y={posterior_edge}")
        
        if not anterior_found:
            anterior_edge = np.percentile(side_points[:, 1], 75)  # Use 75th percentile as anterior edge
            self.logger.warning(f"Using statistical anterior edge at y={anterior_edge}")
        
        # Verify edges are valid (posterior should be less than anterior)
        if posterior_edge >= anterior_edge:
            self.logger.warning("Invalid canal detection, posterior edge is anterior to anterior edge")
            # Swap if necessary
            if posterior_edge > anterior_edge:
                posterior_edge, anterior_edge = anterior_edge, posterior_edge
            # Or apply a small offset if they're equal
            elif posterior_edge == anterior_edge:
                buffer = 5.0  # mm
                posterior_edge -= buffer
                anterior_edge += buffer
        
        # Now search for smallest cross-section within the canal range
        min_area = float('inf')
        min_slice_idx = -1
        min_slice_points = None
        
        # Define slice thickness
        slice_thickness = 2.0  # mm
        
        # Search for the smallest cross-section using the fixed test_x (maintaining side positioning)
        self.logger.info(f"Searching for smallest cross-section between y={posterior_edge} and y={anterior_edge}")
        
        for y_idx in np.arange(posterior_edge, anterior_edge, slice_thickness/2):  # Use smaller steps for better precision
            # Find points in this slice (around test_x for the specific side)
            # Use a cylinder-like approach: check around fixed X, with a slice in Y
            cylinder_points = []
            
            for point in side_points:
                # Check if point is within Y-slice
                if abs(point[1] - y_idx) <= slice_thickness/2:
                    # Check if it's reasonably close to our test_x
                    if abs(point[0] - test_x) <= 15.0:  # 15mm radius in X direction
                        cylinder_points.append(point)
            
            # Calculate slice area (number of points as proxy for area)
            slice_area = len(cylinder_points)
            
            # Skip empty slices or very small samples
            if slice_area < 5:
                continue
            
            # If we have points and area is smaller than current minimum
            if slice_area < min_area:
                min_area = slice_area
                min_slice_idx = y_idx
                min_slice_points = np.array(cylinder_points)
                self.logger.debug(f"New minimum at y={y_idx}, area={slice_area}")
        
        if min_slice_points is None or len(min_slice_points) == 0:
            self.logger.warning("No minimum slice found within pedicle boundaries")
            
            # Try again with a larger slice thickness as fallback
            fallback_thickness = 4.0  # mm
            
            for y_idx in np.arange(posterior_edge, anterior_edge, fallback_thickness):
                # Find points in thicker slice
                cylinder_points = []
                
                for point in side_points:
                    if abs(point[1] - y_idx) <= fallback_thickness/2:
                        if abs(point[0] - test_x) <= 20.0:  # Wider radius too
                            cylinder_points.append(point)
                
                slice_area = len(cylinder_points)
                
                if slice_area >= 5 and (min_area == float('inf') or slice_area < min_area):
                    min_area = slice_area
                    min_slice_idx = y_idx
                    min_slice_points = np.array(cylinder_points)
                    self.logger.debug(f"Fallback: New minimum at y={y_idx}, area={slice_area}")
        
        if min_slice_points is None or len(min_slice_points) == 0:
            self.logger.error("Could not find any valid slice within pedicle")
            return 0, None, float('inf')
        
        self.logger.info(f"Found smallest cross-section at y={min_slice_idx} with area={min_area}")
        
        # Add marker for the smallest slice position
        if hasattr(slicer, 'mrmlScene') and 'debug_fiducials' in locals():
            idx = debug_fiducials.AddFiducial(test_x, min_slice_idx, test_z)
            debug_fiducials.SetNthFiducialLabel(idx, "Smallest Slice")
        
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
        
        # If we don't have any points, return default values
        if min_slice_points is None or len(min_slice_points) == 0:
            self.logger.warning("No points in smallest cross-section")
            return np.zeros(3), None
        
        # Compute the center of the smallest cross-section
        pedicle_center = np.mean(min_slice_points, axis=0)
        
        # Extract the pedicle border from the surface cloud
        pedicle_border_points = None
        if aligned_surface_cloud and aligned_surface_cloud.GetNumberOfPoints() > 0:
            surface_points = self._points_to_numpy(aligned_surface_cloud.GetPoints())
            
            # Find surface points near the slice
            border_mask = np.abs(surface_points[:, 1] - min_slice_idx) < slice_thickness/2
            pedicle_border_points = surface_points[border_mask]
        
        return pedicle_center, pedicle_border_points
    
    def detect_pedicle_center_and_border(self, target_level=None):
        """
        Detect the pedicle center and border using the smallest cross-section approach.
        
        Parameters:
            target_level: Target vertebra level (e.g., 'L4', 'L5')
            
        Returns:
            tuple: (pedicle_center, pedicle_border, pca_vectors) in original space
        """
        import numpy as np
        
        # Step 1: Extract volume and surface point clouds
        volume_cloud, surface_cloud, level_mask = self.extract_volume_and_surface_clouds(target_level)
        
        # Step 2-3: Apply PCA and align point clouds
        aligned_volume_cloud, aligned_surface_cloud, rotation_matrix, center_of_mass, aligned_center, pca_vectors = \
            self.align_point_cloud_with_pca(volume_cloud, surface_cloud, level_mask)
        
        # Step 4: Find smallest cross-section
        # Transform insertion point for side determination
        min_slice_idx, min_slice_points, min_area = self.find_smallest_pedicle_cross_section(
            aligned_volume_cloud, aligned_surface_cloud, aligned_center, 
            center_of_mass, rotation_matrix, self.insertion_point[0])
        
        # Step 5-6: Find pedicle center and border
        aligned_pedicle_center, aligned_pedicle_border = self.compute_pedicle_center_and_border(
            min_slice_points, aligned_surface_cloud, min_slice_idx)
        
        # Step 7: Transform back to original space
        pedicle_center = self.transform_to_original_space(
            aligned_pedicle_center, center_of_mass, rotation_matrix)
        
        pedicle_border = self.transform_to_original_space(
            aligned_pedicle_border, center_of_mass, rotation_matrix)
        
        # Create vtk polydata for pedicle border
        pedicle_border_cloud = self._numpy_to_vtk_polydata(pedicle_border) if pedicle_border is not None else None
        
        # Visualize the results if in Slicer environment
        if hasattr(slicer, 'mrmlScene'):
            # Visualize the smallest cross-section
            if min_slice_points is not None and len(min_slice_points) > 0:
                cross_section_cloud = self._numpy_to_vtk_polydata(min_slice_points)
                cross_section_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"SmallestCrossSection_{target_level}")
                cross_section_model.SetAndObservePolyData(cross_section_cloud)
                cross_section_model.CreateDefaultDisplayNodes()
                if cross_section_model.GetDisplayNode():
                    cross_section_model.GetDisplayNode().SetColor(1.0, 0.0, 1.0)  # Magenta
                    
            # Visualize the pedicle border
            if pedicle_border_cloud and pedicle_border_cloud.GetNumberOfPoints() > 0:
                border_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"PedicleBorder_{target_level}")
                border_model.SetAndObservePolyData(pedicle_border_cloud)
                border_model.CreateDefaultDisplayNodes()
                if border_model.GetDisplayNode():
                    border_model.GetDisplayNode().SetColor(0.0, 1.0, 1.0)  # Cyan
                    
            # Add fiducial for pedicle center
            fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"PedicleCenter_{target_level}")
            fiducial_node.AddFiducial(*pedicle_center)
            fiducial_node.SetNthFiducialLabel(0, "Pedicle Center")
            fiducial_node.CreateDefaultDisplayNodes()
            if fiducial_node.GetDisplayNode():
                fiducial_node.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)  # Orange
        
        return pedicle_center, pedicle_border_cloud, pca_vectors

def visualize_critical_points(vertebra):
    """
    Create visual markers for important points used in trajectory planning.
    
    Parameters:
        vertebra: The Vertebra object containing the critical points
        
    Returns:
        vtkMRMLMarkupsFiducialNode: Debug fiducials node with critical points
    """
    import slicer
    import logging
    import numpy as np
    import vtk
    
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
                
                # Create line source
                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(start_point)
                line_source.SetPoint2(end_point)
                line_source.Update()
                
                # Create or get model node
                line_node_name = f"PCA_Axis_{i+1}"
                line_node = slicer.mrmlScene.GetFirstNodeByName(line_node_name)
                
                if line_node:
                    # If the node exists, remove it first to avoid errors
                    slicer.mrmlScene.RemoveNode(line_node)
                
                # Create a new model node
                line_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", line_node_name)
                line_node.SetAndObservePolyData(line_source.GetOutput())
                
                # Create or set display node
                if not line_node.GetDisplayNode():
                    display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                    line_node.SetAndObserveDisplayNodeID(display_node.GetID())
                
                # Set display properties
                display_node = line_node.GetDisplayNode()
                if display_node:
                    display_node.SetColor(colors[i])
                    display_node.SetLineWidth(3.0)
        
        return debug_fiducials
        
    except Exception as e:
        logging.error(f"Error visualizing critical points: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None