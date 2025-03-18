import numpy as np
import vtk
from vtkmodules.util import numpy_support
from .PCAUtils import apply_pca
import logging
import slicer
import traceback

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
            self.centroid = self._calculate_centroid(self.mask_array)
            self.logger.info(f"Vertebra centroid (RAS): {self.centroid}")
            
            # Convert centroid to IJK for debugging
            centroid_ijk = self._ras_to_ijk(self.centroid)
            self.logger.info(f"Vertebra centroid (IJK): {centroid_ijk}")
            
            # Find spinal canal using the improved method
            canal_min_y, canal_max_y = self._detect_spinal_canal_improved()
            self.logger.info(f"Spinal canal boundaries: min_y={canal_min_y}, max_y={canal_max_y}")
            
            # Cut out pedicle region
            self.pedicle_array = self._cut_pedicle(
                self.masked_volume_array.copy(), 
                canal_max_y, 
                canal_min_y, 
                buffer_front=15,
                buffer_end=1
            )
            
            # Create debug volume for pedicle
            pedicle_debug = self._numpy_to_volume(self.pedicle_array, self.volume_node)
            if pedicle_debug:
                pedicle_debug.SetName("DebugPedicle")
                self.logger.info(f"Pedicle array shape: {self.pedicle_array.shape}, non-zero elements: {np.count_nonzero(self.pedicle_array)}")
            
            # Separate relevant side (left/right based on insertion point)
            self.pedicle_roi_array = self._cut_pedicle_side(
                self.pedicle_array.copy(),
                self.insertion_point[0],
                self.centroid[0]
            )
            
            # Create debug volume for pedicle ROI
            pedicle_roi_debug = self._numpy_to_volume(self.pedicle_roi_array, self.volume_node)
            if pedicle_roi_debug:
                pedicle_roi_debug.SetName("DebugPedicleROI")
                self.logger.info(f"Pedicle ROI array shape: {self.pedicle_roi_array.shape}, non-zero elements: {np.count_nonzero(self.pedicle_roi_array)}")
            
            # Create point cloud for the pedicle
            self.pedicle_point_cloud = self._array_to_point_cloud(
                self.pedicle_roi_array, 
                self.volume_node,
                threshold=0
            )
            
            if self.pedicle_point_cloud:
                num_points = self.pedicle_point_cloud.GetNumberOfPoints()
                self.logger.info(f"Pedicle point cloud has {num_points} points")
                
                # Create a visual model from point cloud for debugging
                if num_points > 0 and hasattr(slicer, 'mrmlScene'):
                    debug_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "DebugPediclePointCloud")
                    debug_model.SetAndObservePolyData(self.pedicle_point_cloud)
                    debug_model.CreateDefaultDisplayNodes()
                    display_node = debug_model.GetDisplayNode()
                    if display_node:
                        display_node.SetColor(1.0, 0.0, 0.0)  # Red
                        display_node.SetRepresentation(display_node.PointsRepresentation)
                        display_node.SetPointSize(5.0)
            
            # Perform PCA on pedicle points
            if self.pedicle_point_cloud and self.pedicle_point_cloud.GetNumberOfPoints() > 0:
                points_array = self._points_to_numpy(self.pedicle_point_cloud.GetPoints())
                self.logger.info(f"Points array shape for PCA: {points_array.shape}")
                
                if points_array.shape[0] >= 3:
                    # Run PCA
                    self.coeff, self.latent, self.score = apply_pca(points_array)
                    self.pedicle_center_point = np.mean(points_array, axis=0)
                    
                    # Scale eigenvectors by eigenvalues for visualization
                    scaling_factor = np.sqrt(self.latent) * 2
                    self.pcaVectors = self.coeff * scaling_factor[:, np.newaxis]
                    self.logger.info(f"PCA principal axis: {self.pcaVectors[:, 2]}")
                    self.logger.info(f"Pedicle center point: {self.pedicle_center_point}")
                    
                    # Create a fiducial for the pedicle center point
                    if hasattr(slicer, 'mrmlScene'):
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
            
            if self.point_cloud:
                num_surface_points = self.point_cloud.GetNumberOfPoints()
                self.logger.info(f"Surface point cloud has {num_surface_points} points")
            
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
        """Create visual models for PCA axes"""
        if not hasattr(slicer, 'mrmlScene'):
            return
        
        if not hasattr(self, 'pcaVectors') or not hasattr(self, 'pedicle_center_point'):
            return
        
        # Colors for the three axes
        axis_colors = [
            (1.0, 0.0, 0.0),  # Red - First axis
            (0.0, 1.0, 0.0),  # Green - Second axis
            (0.0, 0.0, 1.0)   # Blue - Third axis
        ]
        
        # Display length for axes
        display_length = 20.0  # mm
        
        for i in range(3):
            axis_vector = self.pcaVectors[:, i]
            vector_length = np.linalg.norm(axis_vector)
            
            if vector_length < 1e-6:
                continue
                
            # Normalize and scale vector
            axis_vector = axis_vector / vector_length * display_length
            
            # Create line endpoints
            start_point = self.pedicle_center_point - axis_vector/2
            end_point = self.pedicle_center_point + axis_vector/2
            
            # Create line source
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(start_point)
            line_source.SetPoint2(end_point)
            line_source.Update()
            
            # Create model node
            model_name = f"PCA_Axis_{i+1}"
            model_node = slicer.mrmlScene.GetFirstNodeByName(model_name)
            if not model_node:
                model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
            
            model_node.SetAndObservePolyData(line_source.GetOutput())
            
            # Create display node if needed
            if not model_node.GetDisplayNode():
                display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                model_node.SetAndObserveDisplayNodeID(display_node.GetID())
            
            # Set display properties
            display_node = model_node.GetDisplayNode()
            display_node.SetColor(*axis_colors[i])
            display_node.SetLineWidth(3.0)
    
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
            
    def _detect_spinal_canal_improved(self):
        """
        Detect the anterior and posterior boundaries of the spinal canal
        using floodfill-inspired approach similar to the MATLAB implementation.
        
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
            
            # Convert centroid to IJK coordinates
            centroid_ijk = self._ras_to_ijk(self.centroid)
            
            # Get mid-axial slice at centroid Z
            sliceZ = int(np.round(centroid_ijk[2]))
            sliceZ = max(0, min(sliceZ, self.mask_array.shape[2] - 1))
            
            # Extract slice
            slice_mask = self.mask_array[:, :, sliceZ].copy()
            
            # Create a binary mask for the vertebra
            binary_mask = (slice_mask > 0).astype(np.uint8)
            
            # Create a debug volume for visualization
            if hasattr(slicer, 'mrmlScene'):
                # Save binary mask for debugging
                debug_node = self._numpy_to_volume(
                    binary_mask[:, :, np.newaxis],  # Convert 2D slice to 3D volume
                    self.volume_node
                )
                if debug_node:
                    debug_node.SetName("DebugVertebraSlice")
            
            # Find the center point of the vertebra in this slice
            x_idx = int(np.round(centroid_ijk[0]))
            x_idx = max(0, min(x_idx, binary_mask.shape[0] - 1))
            
            # Find a seed point for the spinal canal
            # Start from centroid and move posteriorly until reaching edge
            y_idx = int(np.round(centroid_ijk[1]))
            y_idx = max(0, min(y_idx, binary_mask.shape[1] - 1))
            
            # Store the initial indices for debugging
            initial_x, initial_y = x_idx, y_idx
            
            # 1. Move backward until we exit the vertebra
            posterior_edge = y_idx
            while posterior_edge > 0 and binary_mask[x_idx, posterior_edge] > 0:
                posterior_edge -= 1
                
            self.logger.info(f"Found posterior edge at y={posterior_edge}")
            
            # 2. Move forward from posterior edge until we enter the vertebra again
            anterior_edge = posterior_edge
            while anterior_edge < binary_mask.shape[1] - 1 and binary_mask[x_idx, anterior_edge] == 0:
                anterior_edge += 1
                
            self.logger.info(f"Found anterior edge at y={anterior_edge}")
            
            # If we found a valid canal (gap between vertebra segments)
            if anterior_edge > posterior_edge:
                # Use the floodfill approach to identify the canal more accurately
                try:
                    from scipy.ndimage import binary_fill_holes, label
                    
                    # Seed point in the middle of the suspected canal
                    seed_y = (posterior_edge + anterior_edge) // 2
                    
                    # Create a marker for the seed
                    seed_mask = np.zeros_like(binary_mask)
                    seed_mask[x_idx, seed_y] = 1
                    
                    # Floodfill using binary hole filling
                    # Create inverse mask (0=vertebra, 1=background and canal)
                    inv_mask = 1 - binary_mask
                    
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
                        
                        self.logger.info(f"Found canal boundaries using floodfill: min_y={canal_min_y}, max_y={canal_max_y}")
                        
                        # Add margin to make sure we fully capture the canal
                        canal_min_y = max(0, canal_min_y - 5)
                        canal_max_y = min(binary_mask.shape[1] - 1, canal_max_y + 5)
                        
                        return canal_min_y, canal_max_y
                except Exception as e:
                    self.logger.warning(f"Floodfill method failed: {str(e)}")
                    # Fall back to the simpler method
                    
            # Fallback approach based on edge detection
            # Add margin to make sure we fully capture the canal
            canal_min_y = max(0, posterior_edge - 5)
            canal_max_y = min(binary_mask.shape[1] - 1, anterior_edge + 5)
            
            self.logger.info(f"Found canal boundaries using edge detection: min_y={canal_min_y}, max_y={canal_max_y}")
            
            return canal_min_y, canal_max_y
            
        except Exception as e:
            self.logger.error(f"Error in _detect_spinal_canal_improved: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Default values on error
            canal_min_y = 0
            canal_max_y = self.mask_array.shape[1] // 2
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