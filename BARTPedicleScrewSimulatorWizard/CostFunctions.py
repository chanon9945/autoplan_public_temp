import numpy as np
import vtk
import logging

logger = logging.getLogger(__name__)

def bresenham_3d(origin, endpoint, matrix):
    """
    Draw a 3D line between origin and endpoint in the voxel matrix using Bresenham's algorithm.
    Returns a binary mask where line voxels are 1 and others are 0.
    
    Parameters:
        origin (array): [x, y, z] start point
        endpoint (array): [x, y, z] end point
        matrix (array): 3D numpy array representing the volume
        
    Returns:
        array: Binary mask with same dimensions as matrix
    """
    try:
        # Get matrix dimensions
        matrix_size = matrix.shape
        
        # Round endpoints to integers
        x1, y1, z1 = map(int, np.round(origin))
        x2, y2, z2 = map(int, np.round(endpoint))
        
        # Calculate differences and signs
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        sz = 1 if z2 > z1 else -1
        
        # Determine dominant axis
        if dx >= dy and dx >= dz:
            err_y = 2 * dy - dx
            err_z = 2 * dz - dx
            nsteps = dx
        elif dy >= dx and dy >= dz:
            err_x = 2 * dx - dy
            err_z = 2 * dz - dy
            nsteps = dy
        else:
            err_x = 2 * dx - dz
            err_y = 2 * dy - dz
            nsteps = dz
        
        # Create result matrix
        result = np.zeros(matrix_size, dtype=np.uint8)
        
        # Draw line using Bresenham's algorithm
        x, y, z = x1, y1, z1
        
        for i in range(nsteps + 1):
            # Check if point is within bounds
            if 0 <= x < matrix_size[0] and 0 <= y < matrix_size[1] and 0 <= z < matrix_size[2]:
                result[x, y, z] = 1
                
            # Update coordinates based on dominant axis
            if dx >= dy and dx >= dz:
                if err_y >= 0:
                    y += sy
                    err_y -= 2 * dx
                if err_z >= 0:
                    z += sz
                    err_z -= 2 * dx
                err_y += 2 * dy
                err_z += 2 * dz
                x += sx
            elif dy >= dx and dy >= dz:
                if err_x >= 0:
                    x += sx
                    err_x -= 2 * dy
                if err_z >= 0:
                    z += sz
                    err_z -= 2 * dy
                err_x += 2 * dx
                err_z += 2 * dz
                y += sy
            else:
                if err_x >= 0:
                    x += sx
                    err_x -= 2 * dz
                if err_y >= 0:
                    y += sy
                    err_y -= 2 * dz
                err_x += 2 * dx
                err_y += 2 * dy
                z += sz
        
        return result
        
    except Exception as e:
        logger.error(f"Error in bresenham_3d: {str(e)}")
        # Return empty result on error
        return np.zeros(matrix.shape, dtype=np.uint8)

def find_closest_point_to_line(pt_cloud, line_origin, line_direction):
    """
    Find the closest point in pt_cloud to the line defined by origin and direction.
    
    Parameters:
        pt_cloud (vtkPolyData): Point cloud to search
        line_origin (array): [x, y, z] origin of the line
        line_direction (array): Direction vector of the line
        
    Returns:
        tuple: (closest_point, min_distance)
    """
    try:
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return line_origin, float('inf')
            
        line_direction = line_direction / direction_mag
        
        # Validate point cloud
        if not pt_cloud or not hasattr(pt_cloud, 'GetPoints'):
            logger.warning("Invalid point cloud object")
            return line_origin, float('inf')
            
        points_vtk = pt_cloud.GetPoints()
        if not points_vtk or points_vtk.GetNumberOfPoints() == 0:
            logger.warning("Empty point cloud or no points found")
            return line_origin, float('inf')
            
        # Convert VTK points to numpy array
        n_points = points_vtk.GetNumberOfPoints()
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            points[i] = points_vtk.GetPoint(i)
        
        # Compute vectors from line origin to each point
        vec_to_points = points - line_origin
        
        # Project vectors onto line direction
        proj_scalars = np.sum(vec_to_points * line_direction, axis=1)
        
        # Compute closest points on line
        closest_points = line_origin + proj_scalars[:, np.newaxis] * line_direction
        
        # Compute distances
        distances = np.linalg.norm(points - closest_points, axis=1)
        
        # Find minimum distance and corresponding point
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        closest_point = points[min_idx]
        
        logger.debug(f"Closest point found at distance: {min_distance}")
        return closest_point, min_distance
        
    except Exception as e:
        logger.error(f"Error in find_closest_point_to_line: {str(e)}")
        return line_origin, float('inf')

def point_to_line_distance(point, line_origin, line_direction):
    """
    Calculate the perpendicular distance from a point to a line.
    This function includes detailed debugging to trace the calculation.
    
    Parameters:
        point (array): [x, y, z] point to measure from
        line_origin (array): [x, y, z] origin of the line
        line_direction (array): Direction vector of the line
        
    Returns:
        float: Distance from point to line
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs and convert to numpy arrays
        if point is None or line_origin is None or line_direction is None:
            logger.warning("Invalid inputs to point_to_line_distance")
            return float('inf')
            
        point = np.array(point, dtype=np.float64)
        line_origin = np.array(line_origin, dtype=np.float64)
        line_direction = np.array(line_direction, dtype=np.float64)
        
        # Debug inputs
        logger.debug(f"point_to_line_distance inputs:")
        logger.debug(f"  point: {point}")
        logger.debug(f"  line_origin: {line_origin}")
        logger.debug(f"  line_direction: {line_direction}")
        
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return float('inf')
            
        normalized_direction = line_direction / direction_mag
        logger.debug(f"  normalized_direction: {normalized_direction}")
        
        # Vector from line origin to the point
        vec_to_point = point - line_origin
        logger.debug(f"  vec_to_point: {vec_to_point}")
        
        # Method 1: Cross product
        # The cross product gives a vector perpendicular to both input vectors,
        # with magnitude equal to the area of the parallelogram they form.
        # The perpendicular distance is this area divided by the magnitude of the line direction.
        cross_product = np.cross(vec_to_point, normalized_direction)
        distance1 = np.linalg.norm(cross_product)
        logger.debug(f"  cross_product: {cross_product}")
        logger.debug(f"  distance (cross product method): {distance1}")
        
        # Method 2: Projection
        # Project the vector onto the line direction, then use Pythagoras to find the perpendicular component
        dot_product = np.dot(vec_to_point, normalized_direction)
        projection = normalized_direction * dot_product
        perpendicular_vector = vec_to_point - projection
        distance2 = np.linalg.norm(perpendicular_vector)
        logger.debug(f"  dot_product: {dot_product}")
        logger.debug(f"  projection: {projection}")
        logger.debug(f"  perpendicular_vector: {perpendicular_vector}")
        logger.debug(f"  distance (projection method): {distance2}")
        
        # Method 3: Formula d = |v × w| / |w|
        # Where v is vec_to_point and w is normalized_direction
        # Since w is already normalized, this simplifies to d = |v × w|
        cross_magnitude = np.linalg.norm(np.cross(vec_to_point, normalized_direction))
        distance3 = cross_magnitude
        logger.debug(f"  distance (formula method): {distance3}")
        
        # The results from all methods should be identical (within floating point precision)
        # Return the result from Method 2 (most explicit)
        return distance2
        
    except Exception as e:
        logger.error(f"Error in point_to_line_distance: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return float('inf')

def gen_traj(origin_transform, target_transform):
    """
    Generate trajectory vector from origin to target.
    
    Parameters:
        origin_transform (array): 4x4 homogeneous transform for origin
        target_transform (array): 4x4 homogeneous transform for target
        
    Returns:
        array: Normalized direction vector
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get translation components from transforms
        if origin_transform.shape != (4, 4) or target_transform.shape != (4, 4):
            logger.warning(f"Invalid transform shapes: {origin_transform.shape}, {target_transform.shape}")
            return np.array([0, 1, 0])  # Default direction (anterior)
            
        origin_pos = origin_transform[0:3, 3]
        target_pos = target_transform[0:3, 3]
        
        # Calculate direction vector
        direction = target_pos - origin_pos
        direction_mag = np.linalg.norm(direction)
        
        if direction_mag < 1e-6:
            logger.warning("Generated trajectory is too short")
            return np.array([0, 1, 0])  # Default to anterior direction
            
        return direction / direction_mag
        
    except Exception as e:
        logger.error(f"Error in gen_traj: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return np.array([0, 1, 0])  # Default to anterior direction

def sample_ct_along_trajectory(volume_node, origin, direction, length, num_samples=50):
    """
    Sample CT values along a trajectory line.
    
    Parameters:
        volume_node (vtkMRMLScalarVolumeNode): CT volume node
        origin (array): [x, y, z] start point in RAS
        direction (array): Normalized direction vector
        length (float): Length of trajectory
        num_samples (int): Number of points to sample
        
    Returns:
        array: Array of sampled density values
    """
    try:
        if not volume_node or not hasattr(volume_node, 'GetImageData'):
            logger.warning("Invalid volume node")
            return np.array([])
            
        # Get image data
        image_data = volume_node.GetImageData()
        if not image_data:
            logger.warning("No image data in volume node")
            return np.array([])
            
        # Get RAS to IJK transform
        ras_to_ijk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk)
        
        # Sample points along trajectory
        density_values = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            point_ras = origin + t * direction * length
            
            # Convert RAS to IJK
            point_ijk = np.zeros(4)
            ras_point = np.append(point_ras, 1.0)
            ras_to_ijk.MultiplyPoint(ras_point, point_ijk)
            
            # Convert to integer voxel coordinates
            ijk = [int(round(point_ijk[j])) for j in range(3)]
            
            # Check if point is within volume bounds
            dims = image_data.GetDimensions()
            if (0 <= ijk[0] < dims[0] and 
                0 <= ijk[1] < dims[1] and 
                0 <= ijk[2] < dims[2]):
                
                # Get voxel value (HU)
                value = image_data.GetScalarComponentAsDouble(ijk[0], ijk[1], ijk[2], 0)
                density_values.append(value)
        
        return np.array(density_values)
        
    except Exception as e:
        logger.error(f"Error in sample_ct_along_trajectory: {str(e)}")
        return np.array([])

def compute_safety_margin(trajectory_points, vertebra_model):
    """
    Compute the safety margin between trajectory and vertebra surface.
    
    Parameters:
        trajectory_points (array): Array of points along trajectory
        vertebra_model (vtkPolyData): Surface model of vertebra
        
    Returns:
        float: Minimum distance from trajectory to surface
    """
    try:
        if not vertebra_model or not hasattr(vertebra_model, 'GetPoints'):
            logger.warning("Invalid vertebra model")
            return 0.0
            
        # Create a locator for fast distance queries
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(vertebra_model)
        locator.BuildLocator()
        
        # Find minimum distance from trajectory to surface
        min_distance = float('inf')
        
        for point in trajectory_points:
            # Find closest point
            id = locator.FindClosestPoint(point)
            if id >= 0:
                surface_point = vertebra_model.GetPoint(id)
                distance = np.linalg.norm(np.array(point) - np.array(surface_point))
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            min_distance = 0.0
            
        return min_distance
        
    except Exception as e:
        logger.error(f"Error in compute_safety_margin: {str(e)}")
        return 0.0
    
def distance_cost(insertion_point, trajectory_direction, pedicle_center):
    """
    Calculate only the distance cost component - how close is the trajectory to the pedicle center.
    This function includes detailed debugging information.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        pedicle_center (array): 3D coordinates of pedicle center
        
    Returns:
        float: Distance cost value
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs and convert to numpy arrays
        if insertion_point is None or trajectory_direction is None or pedicle_center is None:
            logger.warning("Invalid inputs to distance_cost")
            return float('inf')
            
        insertion_point = np.array(insertion_point, dtype=np.float64)
        trajectory_direction = np.array(trajectory_direction, dtype=np.float64)
        pedicle_center = np.array(pedicle_center, dtype=np.float64)
        
        # Normalize the trajectory direction
        direction_mag = np.linalg.norm(trajectory_direction)
        if direction_mag < 1e-6:
            logger.warning("Trajectory direction vector is too short")
            return float('inf')
            
        normalized_direction = trajectory_direction / direction_mag
        
        # Debug inputs
        logger.debug(f"distance_cost inputs:")
        logger.debug(f"  insertion_point: {insertion_point}")
        logger.debug(f"  normalized_direction: {normalized_direction}")
        logger.debug(f"  pedicle_center: {pedicle_center}")
        
        # Vector from insertion point to pedicle center
        vec_to_center = pedicle_center - insertion_point
        logger.debug(f"  vec_to_center: {vec_to_center}")
        
        # Project this vector onto the trajectory direction
        dot_product = np.dot(vec_to_center, normalized_direction)
        projection = normalized_direction * dot_product
        logger.debug(f"  dot_product: {dot_product}")
        logger.debug(f"  projection: {projection}")
        
        # The perpendicular component is the difference
        perpendicular_vector = vec_to_center - projection
        logger.debug(f"  perpendicular_vector: {perpendicular_vector}")
        
        # The distance is the magnitude of this perpendicular component
        distance = np.linalg.norm(perpendicular_vector)
        logger.debug(f"  distance: {distance}")
        
        # Calculate the closest point on the trajectory to the pedicle center
        closest_point = insertion_point + projection
        logger.debug(f"  closest_point: {closest_point}")
        
        # If the dot product is negative, the projection is behind the insertion point
        # (pedicle center is behind trajectory origin)
        if dot_product < 0:
            logger.debug("  projection is behind insertion point")
            
        # If distance is greater than a threshold, apply a penalty
        # This helps avoid positions where the trajectory is very far from the pedicle
        threshold = 20.0  # mm
        if distance > threshold:
            # Apply a quadratic penalty for distances beyond the threshold
            penalty = 1.0 + ((distance - threshold) / threshold) ** 2
            distance *= penalty
            logger.debug(f"  applied distance penalty: {penalty}, adjusted distance: {distance}")
        
        return distance
        
    except Exception as e:
        logger.error(f"Error in distance_cost: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return float('inf')

def cost_total(
    insertion_point, 
    trajectory_direction,
    vertebra_model,
    pedicle_axis,
    pedicle_center,
    weights,
    volume_node,
    trajectory_length
):
    """
    Calculate the cost of a trajectory based primarily on distance to pedicle center.
    This function has been simplified to focus only on the distance cost for debugging.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        vertebra_model (vtkPolyData): Surface model of the vertebra (unused in distance-only mode)
        pedicle_axis (array): Principal axis of the pedicle from PCA (unused in distance-only mode)
        pedicle_center (array): 3D coordinates of pedicle center
        weights (array): Weights for different cost components [distance, angle, boundary, density]
        volume_node (vtkMRMLScalarVolumeNode): CT volume node (unused in distance-only mode)
        trajectory_length (float): Maximum length of trajectory (unused in distance-only mode)
        
    Returns:
        tuple: (total_cost, cost_components)
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    cost_components = {"distance": 0.0, "angle": 0.0, "safety": 0.0, "density": 0.0}
    
    # Validate key inputs
    if insertion_point is None or np.any(np.isnan(insertion_point)):
        logger.warning("Invalid insertion point")
        return float('inf'), cost_components
        
    if trajectory_direction is None or np.any(np.isnan(trajectory_direction)):
        logger.warning("Invalid trajectory direction")
        return float('inf'), cost_components
        
    if pedicle_center is None or np.any(np.isnan(pedicle_center)):
        logger.warning("Invalid pedicle center")
        if insertion_point is not None:
            pedicle_center = insertion_point  # Fallback
        else:
            return float('inf'), cost_components
    
    # Calculate distance cost using the enhanced function
    try:
        from .CostFunctions import distance_cost
        cost_components["distance"] = distance_cost(insertion_point, trajectory_direction, pedicle_center)
    except Exception as e:
        logger.error(f"Error importing or calling distance_cost: {str(e)}")
        
        # Fallback direct implementation
        try:
            # Normalize direction
            direction_mag = np.linalg.norm(trajectory_direction)
            if direction_mag < 1e-6:
                logger.warning("Trajectory direction is too short")
                cost_components["distance"] = float('inf')
            else:
                normalized_direction = trajectory_direction / direction_mag
                
                # Calculate perpendicular distance
                vec_to_center = pedicle_center - insertion_point
                projection = np.dot(vec_to_center, normalized_direction) * normalized_direction
                perpendicular_vector = vec_to_center - projection
                cost_components["distance"] = np.linalg.norm(perpendicular_vector)
        except Exception as e2:
            logger.error(f"Error in fallback distance calculation: {str(e2)}")
            cost_components["distance"] = float('inf')
    
    # Log the cost components for debugging
    logger.debug(f"Cost components: {cost_components}")
    
    # Apply weight to distance cost only
    if weights is not None and len(weights) > 0:
        total_cost = weights[0] * cost_components["distance"]
    else:
        total_cost = cost_components["distance"]
    
    logger.debug(f"Total weighted cost: {total_cost}")
    
    return total_cost, cost_components

def visualize_trajectory(insertion_point, trajectory_direction, pedicle_center, trajectory_length=100.0, name_prefix="Debug"):
    """
    Create visualization objects for a trajectory and its relationship to the pedicle center.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        pedicle_center (array): 3D coordinates of pedicle center
        trajectory_length (float): Length of the trajectory line
        name_prefix (str): Prefix for node names
        
    Returns:
        tuple: (trajectory_node, distance_line_node, point_node) - MRML nodes for visualization
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate the trajectory end point
        end_point = insertion_point + trajectory_direction * trajectory_length
        
        # Create a line source for the trajectory
        trajectory_line = vtk.vtkLineSource()
        trajectory_line.SetPoint1(insertion_point)
        trajectory_line.SetPoint2(end_point)
        trajectory_line.Update()
        
        # Create a model node for the trajectory
        trajectory_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Trajectory")
        trajectory_node.SetAndObservePolyData(trajectory_line.GetOutput())
        
        # Create a display node for the trajectory
        trajectory_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        trajectory_display.SetColor(0.0, 1.0, 0.0)  # Green
        trajectory_display.SetLineWidth(3.0)
        trajectory_node.SetAndObserveDisplayNodeID(trajectory_display.GetID())
        
        # Calculate the closest point on the trajectory to the pedicle center
        # Project pedicle_center onto the trajectory line
        vec_to_center = pedicle_center - insertion_point
        projection = np.dot(vec_to_center, trajectory_direction) * trajectory_direction
        closest_point = insertion_point + projection
        
        # Create a line source for the distance line
        distance_line = vtk.vtkLineSource()
        distance_line.SetPoint1(pedicle_center)
        distance_line.SetPoint2(closest_point)
        distance_line.Update()
        
        # Create a model node for the distance line
        distance_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Distance")
        distance_node.SetAndObservePolyData(distance_line.GetOutput())
        
        # Create a display node for the distance line
        distance_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        distance_display.SetColor(1.0, 0.0, 0.0)  # Red
        distance_display.SetLineWidth(2.0)
        distance_node.SetAndObserveDisplayNodeID(distance_display.GetID())
        
        # Create a point for the pedicle center
        point_source = vtk.vtkSphereSource()
        point_source.SetCenter(pedicle_center)
        point_source.SetRadius(2.0)
        point_source.Update()
        
        # Create a model node for the point
        point_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name_prefix}_Center")
        point_node.SetAndObservePolyData(point_source.GetOutput())
        
        # Create a display node for the point
        point_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        point_display.SetColor(1.0, 1.0, 0.0)  # Yellow
        point_node.SetAndObserveDisplayNodeID(point_display.GetID())
        
        # Calculate and log the distance
        distance = np.linalg.norm(pedicle_center - closest_point)
        logger.info(f"Distance from trajectory to pedicle center: {distance:.2f} mm")
        
        return trajectory_node, distance_node, point_node
        
    except Exception as e:
        logger.error(f"Error in visualize_trajectory: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def visualize_search_result(vertebra, insertion_point, final_traj, angles, cost, name_prefix="Optimal"):
    """
    Create visualization of the final search result.
    
    Parameters:
        vertebra: Vertebra object containing anatomical information
        insertion_point: 3D coordinates of insertion point
        final_traj: Vector representing the optimal trajectory direction
        angles: Tuple of (vertical_angle, horizontal_angle) in degrees
        cost: Cost value of the final trajectory
        name_prefix: Prefix for node names
        
    Returns:
        tuple: (trajectory_node, distance_node, point_node, info_node) - MRML nodes for visualization
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create basic trajectory visualization
        trajectory_node, distance_node, point_node = visualize_trajectory(
            insertion_point, 
            final_traj, 
            vertebra.pedicle_center_point if hasattr(vertebra, 'pedicle_center_point') else vertebra.centroid, 
            trajectory_length=100.0, 
            name_prefix=name_prefix
        )
        
        # Create a text annotation for the result
        vertical_angle, horizontal_angle = angles
        info_text = f"Trajectory Info:\n"
        info_text += f"Vertical Angle: {vertical_angle:.1f}°\n"
        info_text += f"Horizontal Angle: {horizontal_angle:.1f}°\n"
        info_text += f"Cost: {cost:.2f}\n"
        
        info_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", f"{name_prefix}_Info")
        info_node.SetText(info_text)
        
        # Log the result
        logger.info(f"\n{info_text}")
        
        return trajectory_node, distance_node, point_node, info_node
        
    except Exception as e:
        logger.error(f"Error in visualize_search_result: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None
    
def test_distance_cost_function():
    """
    Test and validate the distance cost function with controlled inputs.
    Creates a simple test case and visualizes the results.
    
    This function can be called from the Python console in Slicer to debug the cost function:
        from BARTPedicleScrewSimulatorWizard.CostFunctions import test_distance_cost_function
        test_distance_cost_function()
    """
    import slicer
    import vtk
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output
    
    # Create console handler if not already present
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    logger.info("Running distance cost function test...")
    
    # Create test case with known geometry
    # Define a line that passes 5 units away from the origin
    insertion_point = np.array([0.0, 0.0, 0.0])
    trajectory_direction = np.array([0.0, 1.0, 0.0])  # Pointing along y-axis
    pedicle_center = np.array([5.0, 10.0, 0.0])  # 5 units to the right, 10 units forward
    
    # Expected distance: 5.0 (x-component of pedicle_center)
    expected_distance = 5.0
    
    # Calculate distance using our function
    try:
        from .CostFunctions import distance_cost, point_to_line_distance
        
        # Test distance_cost function
        distance = distance_cost(insertion_point, trajectory_direction, pedicle_center)
        logger.info(f"distance_cost result: {distance}")
        logger.info(f"Expected result: {expected_distance}")
        logger.info(f"Difference: {abs(distance - expected_distance)}")
        
        # Test point_to_line_distance function as well
        distance2 = point_to_line_distance(pedicle_center, insertion_point, trajectory_direction)
        logger.info(f"point_to_line_distance result: {distance2}")
        logger.info(f"Expected result: {expected_distance}")
        logger.info(f"Difference: {abs(distance2 - expected_distance)}")
        
    except Exception as e:
        logger.error(f"Error during cost function test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
    # Visualize the test case
    try:
        # Create a line source for the trajectory
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(insertion_point)
        end_point = insertion_point + trajectory_direction * 20.0  # 20 units long
        line_source.SetPoint2(end_point)
        line_source.Update()
        
        # Create model for trajectory
        trajectory_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestTrajectory")
        trajectory_model.SetAndObservePolyData(line_source.GetOutput())
        
        # Create display node for trajectory
        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        display_node.SetColor(0.0, 1.0, 0.0)  # Green
        display_node.SetLineWidth(2.0)
        trajectory_model.SetAndObserveDisplayNodeID(display_node.GetID())
        
        # Create a sphere source for pedicle center
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(pedicle_center)
        sphere_source.SetRadius(1.0)
        sphere_source.Update()
        
        # Create model for pedicle center
        center_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestPedicleCenter")
        center_model.SetAndObservePolyData(sphere_source.GetOutput())
        
        # Create display node for pedicle center
        center_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        center_display.SetColor(1.0, 0.0, 0.0)  # Red
        center_model.SetAndObserveDisplayNodeID(center_display.GetID())
        
        # Calculate closest point on trajectory to pedicle center
        normalized_direction = trajectory_direction / np.linalg.norm(trajectory_direction)
        vec_to_center = pedicle_center - insertion_point
        projection = np.dot(vec_to_center, normalized_direction) * normalized_direction
        closest_point = insertion_point + projection
        
        # Create line from pedicle center to closest point (perpendicular distance)
        distance_line = vtk.vtkLineSource()
        distance_line.SetPoint1(pedicle_center)
        distance_line.SetPoint2(closest_point)
        distance_line.Update()
        
        # Create model for distance line
        distance_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestDistanceLine")
        distance_model.SetAndObservePolyData(distance_line.GetOutput())
        
        # Create display node for distance line
        distance_display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        distance_display.SetColor(1.0, 1.0, 0.0)  # Yellow
        distance_display.SetLineWidth(2.0)
        distance_model.SetAndObserveDisplayNodeID(distance_display.GetID())
        
        # Change to 3D view
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        
        # Create a text annotation with the test results
        test_info = (
            f"Distance Cost Function Test\n"
            f"----------------------------\n"
            f"Insertion point: {insertion_point}\n"
            f"Trajectory direction: {trajectory_direction}\n"
            f"Pedicle center: {pedicle_center}\n"
            f"Expected distance: {expected_distance}\n"
            f"Calculated distance: {distance}\n"
            f"Difference: {abs(distance - expected_distance)}\n"
        )
        
        # Create a text node with the test info
        text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", "TestDistanceCostInfo")
        text_node.SetText(test_info)
        
        # Display the text in the Python console
        logger.info("\n" + test_info)
        
        return {
            "trajectory_model": trajectory_model,
            "center_model": center_model,
            "distance_model": distance_model,
            "text_node": text_node,
            "distance": distance,
            "expected_distance": expected_distance
        }
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")