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

def cost_total(origin, origin_transform, transform, pt_cloud, pc_vector, 
              pedicle_center_point, weight, resampled_vol, reach):
    """
    Calculate total cost for a trajectory
    
    Parameters:
        origin: 3D position of trajectory origin
        origin_transform: Homogeneous transform at origin
        transform: Homogeneous transform at target
        pt_cloud: Point cloud of vertebra surface
        pc_vector: Principal component vector (from PCA)
        pedicle_center_point: 3D position of pedicle center
        weight: Weight factors [distance_weight, angle_weight, boundary_weight, density_weight]
        resampled_vol: Voxel array of CT intensities
        reach: Maximum reach distance
        
    Returns:
        total_cost: Weighted sum of all cost components
    """
    try:
        # Generate trajectory
        direction = gen_traj(origin_transform, transform)
        
        # Calculate line end point
        line_end = origin + direction * reach
        
        # Create line voxelization
        try:
            if isinstance(resampled_vol, vtk.vtkImageData):
                # Convert VTK image to numpy array
                from vtkmodules.util import numpy_support
                dims = resampled_vol.GetDimensions()
                scalar_data = resampled_vol.GetPointData().GetScalars()
                numpy_array = numpy_support.vtk_to_numpy(scalar_data)
                numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)
                line_voxel = bresenham_3d(origin, line_end, numpy_array)
                cost_density = np.sum(line_voxel * numpy_array)
            else:
                # Direct numpy array
                line_voxel = bresenham_3d(origin, line_end, resampled_vol)
                cost_density = np.sum(line_voxel * resampled_vol)
        except Exception as e:
            logger.error(f"Error in line voxelization: {str(e)}")
            cost_density = 0
        
        # Calculate cost components
        
        # 1. Boundary cost - find closest distance to surface
        try:
            _, cost_boundary = find_closest_point_to_line(pt_cloud, origin, direction)
            if cost_boundary == float('inf'):
                cost_boundary = reach  # Use maximum distance if computation fails
        except Exception as e:
            logger.error(f"Error in boundary cost calculation: {str(e)}")
            cost_boundary = reach
        
        # 2. Angle cost - alignment with principal component
        try:
            pc_norm = np.linalg.norm(pc_vector)
            if pc_norm < 1e-6:
                cost_angle = 0  # No preferred direction
            else:
                pc_normalized = pc_vector / pc_norm
                cost_angle = np.arccos(np.clip(np.abs(np.dot(direction, pc_normalized)), 0, 1))
        except Exception as e:
            logger.error(f"Error in angle cost calculation: {str(e)}")
            cost_angle = 0
        
        # 3. Distance cost - distance from pedicle center
        try:
            cost_dist = point_to_line_distance(pedicle_center_point, origin, direction)
            if cost_dist == float('inf'):
                cost_dist = reach  # Use maximum distance if computation fails
        except Exception as e:
            logger.error(f"Error in distance cost calculation: {str(e)}")
            cost_dist = reach
        
        # Total weighted cost
        total_cost = (
            weight[0] * cost_dist + 
            weight[1] * cost_angle + 
            (1.0/weight[2]) * cost_boundary + 
            weight[3] * cost_density
        )
        
        return total_cost
        
    except Exception as e:
        logger.error(f"Error in cost_total: {str(e)}")
        return float('inf')  # Return maximum cost on error