import numpy as np
import vtk
import logging

logger = logging.getLogger(__name__)

def bresenham_3d(origin, endpoint, matrix):
    """
    Draw a 3D line between origin and endpoint in the voxel matrix using Bresenham's algorithm
    Returns a binary mask where line voxels are 1 and others are 0
    """
    try:
        # Get matrix dimensions
        matrix_size = matrix.shape
        
        # Round endpoints to integers
        x1, y1, z1 = map(round, origin)
        x2, y2, z2 = map(round, endpoint)
        
        # Calculate differences and signs
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        sz = 1 if z2 > z1 else -1
        
        # Determine dominant axis
        if dx >= dy and dx >= dz:
            nsteps = dx + 1
        elif dy >= dx and dy >= dz:
            nsteps = dy + 1
        else:
            nsteps = dz + 1
        
        # Create result matrix
        result = np.zeros(matrix_size, dtype=np.uint8)
        
        # Draw line using Bresenham's algorithm
        x, y, z = x1, y1, z1
        
        # Error terms
        if dx >= dy and dx >= dz:
            err1 = 2*dy - dx
            err2 = 2*dz - dx
            
            for i in range(nsteps):
                # Set voxel if within bounds
                if 0 <= x < matrix_size[0] and 0 <= y < matrix_size[1] and 0 <= z < matrix_size[2]:
                    result[x, y, z] = 1
                    
                # Update coordinates
                if err1 > 0:
                    y += sy
                    err1 -= 2*dx
                if err2 > 0:
                    z += sz
                    err2 -= 2*dx
                
                err1 += 2*dy
                err2 += 2*dz
                x += sx
                
        elif dy >= dx and dy >= dz:
            err1 = 2*dx - dy
            err2 = 2*dz - dy
            
            for i in range(nsteps):
                # Set voxel if within bounds
                if 0 <= x < matrix_size[0] and 0 <= y < matrix_size[1] and 0 <= z < matrix_size[2]:
                    result[x, y, z] = 1
                    
                # Update coordinates
                if err1 > 0:
                    x += sx
                    err1 -= 2*dy
                if err2 > 0:
                    z += sz
                    err2 -= 2*dy
                
                err1 += 2*dx
                err2 += 2*dz
                y += sy
                
        else:
            err1 = 2*dx - dz
            err2 = 2*dy - dz
            
            for i in range(nsteps):
                # Set voxel if within bounds
                if 0 <= x < matrix_size[0] and 0 <= y < matrix_size[1] and 0 <= z < matrix_size[2]:
                    result[x, y, z] = 1
                    
                # Update coordinates
                if err1 > 0:
                    x += sx
                    err1 -= 2*dz
                if err2 > 0:
                    y += sy
                    err2 -= 2*dz
                
                err1 += 2*dx
                err2 += 2*dy
                z += sz
        
        return result
        
    except Exception as e:
        logger.error(f"Error in bresenham_3d: {str(e)}")
        # Return empty result on error
        return np.zeros(matrix.shape, dtype=np.uint8)

def find_closest_point_to_line(pt_cloud, line_origin, line_direction):
    """
    Find the closest point in pt_cloud to the line defined by origin and direction
    Returns the closest point and its distance to the line
    """
    try:
        # Normalize direction vector
        direction_mag = np.linalg.norm(line_direction)
        if direction_mag < 1e-6:
            logger.warning("Line direction vector is too short")
            return line_origin, float('inf')
            
        line_direction = line_direction / direction_mag
        
        # Get point cloud data
        if not pt_cloud or not hasattr(pt_cloud, 'GetPoints'):
            logger.warning("Invalid point cloud object")
            return line_origin, float('inf')
            
        points_vtk = pt_cloud.GetPoints()
        if not points_vtk or points_vtk.GetNumberOfPoints() == 0:
            logger.warning("Empty point cloud")
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
        
        return closest_point, min_distance
        
    except Exception as e:
        logger.error(f"Error in find_closest_point_to_line: {str(e)}")
        return line_origin, float('inf')

def point_to_line_distance(point, line_origin, line_direction):
    """
    Calculate the perpendicular distance from a point to a line
    """
    try:
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
    Generate trajectory vector from origin to target
    """
    try:
        origin_pos = origin_transform[0:3, 3]
        target_pos = target_transform[0:3, 3]
        
        direction = target_pos - origin_pos
        direction_mag = np.linalg.norm(direction)
        
        if direction_mag < 1e-6:
            logger.warning("Generated trajectory is too short")
            return np.array([0, 1, 0])  # Default to anterior direction
            
        return direction / direction_mag
        
    except Exception as e:
        logger.error(f"Error in gen_traj: {str(e)}")
        return np.array([0, 1, 0])  # Default to anterior direction

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
        float: Combined weighted cost (lower is better)
    """
    cost_components = {}
    
    # 1. Cost based on alignment with pedicle axis
    # Higher alignment = lower cost
    pedicle_axis_norm = np.linalg.norm(pedicle_axis)
    if pedicle_axis_norm > 1e-6:
        pedicle_axis = pedicle_axis / pedicle_axis_norm
        
    angle_cost = np.arccos(np.abs(np.dot(trajectory_direction, pedicle_axis)))
    cost_components['angle'] = angle_cost
    
    # 2. Cost based on distance from pedicle center
    # Calculate distance from pedicle center to trajectory line
    center_to_insertion = pedicle_center - insertion_point
    projection = np.dot(center_to_insertion, trajectory_direction)
    closest_point = insertion_point + projection * trajectory_direction
    distance_cost = np.linalg.norm(pedicle_center - closest_point)
    cost_components['distance'] = distance_cost
    
    # 3. Cost based on safety margin from vertebra surface
    # Use vtkDistancePolyDataFilter to find closest distance
    if vertebra_model and hasattr(vertebra_model, 'GetPoints') and vertebra_model.GetPoints():
        # Create a line representation
        line_points = vtk.vtkPoints()
        line_points.InsertNextPoint(insertion_point)
        end_point = insertion_point + trajectory_direction * trajectory_length
        line_points.InsertNextPoint(end_point)
        
        line_cells = vtk.vtkCellArray()
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        line_cells.InsertNextCell(line)
        
        trajectory_polydata = vtk.vtkPolyData()
        trajectory_polydata.SetPoints(line_points)
        trajectory_polydata.SetLines(line_cells)
        
        # Use distance filter
        distance_filter = vtk.vtkDistancePolyDataFilter()
        distance_filter.SetInputData(0, trajectory_polydata)
        distance_filter.SetInputData(1, vertebra_model)
        distance_filter.SignedDistanceOff()
        distance_filter.Update()
        
        # Get minimum distance
        output = distance_filter.GetOutput()
        distances = vtk.util.numpy_support.vtk_to_numpy(
            output.GetPointData().GetArray('Distance')
        )
        safety_cost = np.min(distances) if distances.size > 0 else trajectory_length
        
        # Invert: smaller distance = higher cost
        if safety_cost < 1.0:  # If too close to surface
            safety_cost = 1.0 / max(safety_cost, 0.1)  # Avoid division by zero
        else:
            safety_cost = 0.0  # Safe distance
    else:
        safety_cost = 0.0
    cost_components['safety'] = safety_cost
    
    # 4. Cost based on bone density along trajectory
    # Sample the CT volume along the trajectory
    if volume_node:
        # Get image data
        image_data = volume_node.GetImageData()
        
        # Get RAS to IJK transform
        ras_to_ijk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk)
        
        # Sample points along trajectory
        num_samples = 50
        density_values = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            point_ras = insertion_point + t * trajectory_direction * trajectory_length
            
            # Convert RAS to IJK
            point_ijk = [0, 0, 0, 1]
            ras_to_ijk.MultiplyPoint(np.append(point_ras, 1.0), point_ijk)
            
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
        
        # Calculate mean density along trajectory
        if density_values:
            # Define target range for bone (e.g., trabecular bone is ~200-400 HU)
            min_target_hu = 200
            max_target_hu = 400
            
            # Convert to numpy array for easier calculation
            density_array = np.array(density_values)
            
            # Count voxels in target range (good density)
            good_density_count = np.sum((density_array >= min_target_hu) & 
                                       (density_array <= max_target_hu))
            
            # Calculate percentage of trajectory in good bone
            good_density_ratio = good_density_count / len(density_values)
            
            # Invert: higher good density ratio = lower cost
            density_cost = 1.0 - good_density_ratio
        else:
            density_cost = 1.0  # Maximum cost if no samples
    else:
        density_cost = 0.0
    cost_components['density'] = density_cost
    
    # Calculate weighted sum
    total_cost = (
        weights[0] * cost_components['distance'] +
        weights[1] * cost_components['angle'] +
        weights[2] * cost_components['safety'] +
        weights[3] * cost_components['density']
    )
    
    return total_cost, cost_components