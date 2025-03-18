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
    
    Parameters:
        point (array): [x, y, z] point to measure from
        line_origin (array): [x, y, z] origin of the line
        line_direction (array): Direction vector of the line
        
    Returns:
        float: Distance from point to line
    """
    try:
        # Validate inputs
        if point is None or line_origin is None or line_direction is None:
            logger.warning("Invalid inputs to point_to_line_distance")
            return float('inf')
            
        # Convert to numpy arrays if not already
        point = np.array(point)
        line_origin = np.array(line_origin)
        line_direction = np.array(line_direction)
        
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return float('inf')
            
        line_direction = line_direction / direction_mag
        
        # Vector from origin to point
        vec_to_point = point - line_origin
        
        # Cross product gives perpendicular distance
        cross_product = np.cross(vec_to_point, line_direction)
        
        return np.linalg.norm(cross_product)
        
    except Exception as e:
        logger.error(f"Error in point_to_line_distance: {str(e)}")
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
    try:
        # Get translation components from transforms
        if origin_transform.shape != (4, 4) or target_transform.shape != (4, 4):
            logger.warning(f"Invalid transform shapes: {origin_transform.shape}, {target_transform.shape}")
            return np.array([0, 1, 0])  # Default direction
            
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
    Calculate the cost of a trajectory based on multiple criteria.
    
    Parameters:
        insertion_point (array): 3D coordinates of insertion point
        trajectory_direction (array): Unit vector of trajectory direction
        vertebra_model (vtkPolyData): Surface model of the vertebra
        pedicle_axis (array): Principal axis of the pedicle from PCA
        pedicle_center (array): 3D coordinates of pedicle center
        weights (array): Weights for different cost components [distance, angle, boundary, density]
        volume_node (vtkMRMLScalarVolumeNode): CT volume node
        trajectory_length (float): Maximum length of trajectory
        
    Returns:
        tuple: (total_cost, cost_components)
    """
    cost_components = {}
    
    # Validate inputs
    if insertion_point is None or np.any(np.isnan(insertion_point)):
        logger.warning("Invalid insertion point")
        return float('inf'), {"error": "Invalid insertion point"}
        
    if trajectory_direction is None or np.any(np.isnan(trajectory_direction)):
        logger.warning("Invalid trajectory direction")
        return float('inf'), {"error": "Invalid trajectory direction"}
        
    if pedicle_center is None or np.any(np.isnan(pedicle_center)):
        logger.warning("Invalid pedicle center")
        pedicle_center = insertion_point  # Fallback
    
    # 1. Cost based on alignment with pedicle axis
    # Higher alignment = lower cost
    try:
        pedicle_axis_norm = np.linalg.norm(pedicle_axis)
        if pedicle_axis_norm > 1e-6:
            pedicle_axis = pedicle_axis / pedicle_axis_norm
            # Use absolute value of dot product to handle opposite directions
            alignment = np.abs(np.dot(trajectory_direction, pedicle_axis))
            angle_cost = np.arccos(np.clip(alignment, -1.0, 1.0))
        else:
            logger.warning("Pedicle axis is too short")
            angle_cost = np.pi/2  # 90 degrees = worst case
    except Exception as e:
        logger.error(f"Error calculating angle cost: {str(e)}")
        angle_cost = np.pi/2
        
    cost_components['angle'] = angle_cost
    
    # 2. Cost based on distance from pedicle center
    try:
        distance_cost = point_to_line_distance(pedicle_center, insertion_point, trajectory_direction)
    except Exception as e:
        logger.error(f"Error calculating distance cost: {str(e)}")
        distance_cost = trajectory_length  # Worst case
        
    cost_components['distance'] = distance_cost
    
    # 3. Cost based on safety margin from vertebra surface
    safety_cost = 0.0
    
    if vertebra_model and hasattr(vertebra_model, 'GetPoints') and vertebra_model.GetPoints():
        # Check if the vertebra model actually has points
        if vertebra_model.GetNumberOfPoints() > 0:
            try:
                # Sample points along trajectory
                num_points = 20
                trajectory_points = []
                for i in range(num_points):
                    t = i / (num_points - 1)
                    point = insertion_point + t * trajectory_direction * trajectory_length
                    trajectory_points.append(point)
                    
                # Create trajectory polydata
                points_vtk = vtk.vtkPoints()
                for point in trajectory_points:
                    points_vtk.InsertNextPoint(point)
                    
                line_cells = vtk.vtkCellArray()
                for i in range(len(trajectory_points) - 1):
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, i + 1)
                    line_cells.InsertNextCell(line)
                    
                traj_polydata = vtk.vtkPolyData()
                traj_polydata.SetPoints(points_vtk)
                traj_polydata.SetLines(line_cells)
                
                # Make sure both trajectory and vertebra model have points
                if traj_polydata.GetNumberOfPoints() > 0 and vertebra_model.GetNumberOfPoints() > 0:
                    # Use a simpler approach with vtkPointLocator instead of vtkDistancePolyDataFilter
                    # since the filter can have issues with empty datasets
                    locator = vtk.vtkPointLocator()
                    locator.SetDataSet(vertebra_model)
                    locator.BuildLocator()
                    
                    min_distance = float('inf')
                    for point in trajectory_points:
                        id = locator.FindClosestPoint(point)
                        if id >= 0:
                            closest_point = vertebra_model.GetPoint(id)
                            distance = np.linalg.norm(np.array(point) - np.array(closest_point))
                            min_distance = min(min_distance, distance)
                    
                    if min_distance != float('inf'):
                        # Invert: smaller distance = higher cost (with safety threshold)
                        safety_threshold = 1.0  # mm
                        if min_distance < safety_threshold:
                            safety_cost = safety_threshold / max(min_distance, 0.1)  # Avoid division by zero
                        else:
                            safety_cost = 0.0  # Safe distance
                    else:
                        safety_cost = 0.0
                else:
                    logger.warning("Trajectory or vertebra model has no points")
            except Exception as e:
                logger.error(f"Error calculating safety cost: {str(e)}")
                safety_cost = 0.0  # Default to neutral cost
        else:
            logger.warning("Vertebra model has no points")
    
    cost_components['safety'] = safety_cost
    
    # 4. Cost based on bone density along trajectory
    density_cost = 0.0
    if volume_node:
        try:
            # Sample CT along trajectory
            density_values = sample_ct_along_trajectory(
                volume_node, insertion_point, trajectory_direction, trajectory_length)
            
            if len(density_values) > 0:
                # Define target HU ranges for different bone types
                soft_tissue_range = (-100, 135)  # Avoid
                cancellous_bone_range = (135, 375)  # Preferred
                cortical_bone_range = (375, 1200)  # OK in moderation
                
                # Calculate percentage in each range
                n_samples = len(density_values)
                soft_tissue_count = np.sum((density_values >= soft_tissue_range[0]) & 
                                          (density_values < soft_tissue_range[1]))
                cancellous_count = np.sum((density_values >= cancellous_bone_range[0]) & 
                                         (density_values < cancellous_bone_range[1]))
                cortical_count = np.sum((density_values >= cortical_bone_range[0]) & 
                                       (density_values < cortical_bone_range[1]))
                
                # Calculate density score (higher is better)
                if n_samples > 0:
                    # Preferred: high percentage of cancellous bone, limited cortical, minimal soft tissue
                    cancellous_ratio = cancellous_count / n_samples
                    cortical_ratio = cortical_count / n_samples
                    soft_tissue_ratio = soft_tissue_count / n_samples
                    
                    # Density score (0 to 1, higher is better)
                    density_score = (
                        1.0 * cancellous_ratio +  # Full weight for cancellous
                        0.5 * cortical_ratio -    # Partial weight for cortical
                        1.0 * soft_tissue_ratio   # Penalty for soft tissue
                    )
                    
                    # Constrain to 0-1 range and invert (lower is better for cost)
                    density_score = np.clip(density_score, 0.0, 1.0)
                    density_cost = 1.0 - density_score
                else:
                    density_cost = 1.0  # Maximum cost if no samples
            else:
                density_cost = 1.0  # Maximum cost if no density values
        except Exception as e:
            logger.error(f"Error calculating density cost: {str(e)}")
            density_cost = 0.5  # Neutral cost
    
    cost_components['density'] = density_cost
    
    # Calculate weighted sum
    try:
        total_cost = (
            weights[0] * cost_components['distance'] +
            weights[1] * cost_components['angle'] +
            weights[2] * cost_components['safety'] +
            weights[3] * cost_components['density']
        )
    except Exception as e:
        logger.error(f"Error calculating total cost: {str(e)}")
        total_cost = float('inf')
    
    logger.debug(f"Cost components: {cost_components}, Total: {total_cost}")
    return total_cost, cost_components