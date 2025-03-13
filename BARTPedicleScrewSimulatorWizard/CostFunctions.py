import numpy as np
import vtk

def bresenham_3d(origin, endpoint, matrix):
    """
    Draw a 3D line between origin and endpoint in the voxel matrix using Bresenham's algorithm
    Returns a binary mask where line voxels are 1 and others are 0
    """
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

def find_closest_point_to_line(pt_cloud, line_origin, line_direction):
    """
    Find the closest point in pt_cloud to the line defined by origin and direction
    Returns the closest point and its distance to the line
    """
    # Normalize direction vector
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # Get point cloud locations
    points = np.array(pt_cloud.GetPoints().GetData())
    points = points.reshape(-1, 3)
    
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

def point_to_line_distance(point, line_origin, line_direction):
    """
    Calculate the perpendicular distance from a point to a line
    """
    # Normalize direction vector
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # Vector from origin to point
    vec_to_point = point - line_origin
    
    # Cross product gives perpendicular distance
    cross_product = np.cross(vec_to_point, line_direction)
    
    return np.linalg.norm(cross_product)

def gen_traj(origin_transform, target_transform):
    """
    Generate trajectory vector from origin to target
    """
    origin_pos = origin_transform[0:3, 3]
    target_pos = target_transform[0:3, 3]
    
    direction = target_pos - origin_pos
    return direction / np.linalg.norm(direction)

def cost_total(origin, origin_transform, transform, pt_cloud, pc_vector, 
              pedicle_center_point, weight, resampled_vol, reach):
    """
    Calculate total cost for a trajectory
    """
    # Generate trajectory
    direction = gen_traj(origin_transform, transform)
    
    # Calculate line end point
    line_end = origin + direction * reach
    
    # Create line voxelization
    line_voxel = bresenham_3d(origin, line_end, resampled_vol)
    
    # Calculate cost components
    
    # 1. Boundary cost - find closest distance to surface
    _, cost_boundary = find_closest_point_to_line(pt_cloud, origin, direction)
    
    # 2. Angle cost - alignment with principal component
    cost_angle = np.arctan2(
        np.linalg.norm(np.cross(direction, pc_vector)),
        np.dot(direction, pc_vector)
    )
    
    # 3. Distance cost - distance from pedicle center
    cost_dist = point_to_line_distance(pedicle_center_point, origin, direction)
    
    # 4. Density cost - sum of intensity along line
    cost_density = np.sum(line_voxel * resampled_vol)
    
    # Total weighted cost
    total_cost = (
        weight[0] * cost_dist + 
        weight[1] * cost_angle + 
        (1.0/weight[2]) * cost_boundary + 
        weight[3] * cost_density
    )
    
    return total_cost