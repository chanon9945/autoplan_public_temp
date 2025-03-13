import numpy as np
from .RobotInit import Robot
from .CostFunctions import cost_total, gen_traj

class PedicleScrewAutoPlanner:
    """
    Automatic trajectory planning for pedicle screws
    """
    def __init__(self, resolution=1000, reach=100, weight=None):
        if weight is None:
            weight = [300, 1, 30, 0.05]
        
        self.resolution = resolution
        self.reach = reach
        self.weight = weight
        
        # Initialize robot workspace
        self.robot = Robot(resolution, reach)
        
        # Calculate mean transforms
        self.h1_mean, self.h2_mean = self.robot.get_mean_transforms()
    
    def plan_trajectory(self, vertebra, insertion_point):
        """
        Plan optimal trajectory for a given vertebra and insertion point
        """
        # Create insertion transform
        insertion_transform = np.eye(4)
        insertion_transform[0:3, 3] = insertion_point
        
        # Get pedicle center point
        if hasattr(vertebra.pedicle_point_cloud, 'Location'):
            pedicle_center = np.mean(vertebra.pedicle_point_cloud.Location, axis=0)
        else:
            # For VTK point clouds
            points = vertebra.pedicle_point_cloud.GetPoints()
            n_points = points.GetNumberOfPoints()
            point_sum = np.zeros(3)
            for i in range(n_points):
                point = points.GetPoint(i)
                point_sum += np.array(point)
            pedicle_center = point_sum / n_points
        
        # Run optimization
        final_traj, cost = self.optimize_trajectory(
            insertion_point,
            insertion_transform,
            vertebra.point_cloud,
            vertebra.pcaVectors[:, 2], # Use 3rd principal component
            pedicle_center,
            vertebra.maskedVolume
        )
        
        # Calculate angles for UI
        vertical_angle, horizontal_angle = self.calculate_angles(final_traj)
        
        return final_traj, (vertical_angle, horizontal_angle), cost
    
    def optimize_trajectory(self, insertion_point, insertion_transform, 
                           point_cloud, pca_vector, pedicle_center, volume):
        """
        Find optimal trajectory by searching robot workspace
        """
        # Phase 1: Search in first joint space
        cost_h1 = np.zeros(self.resolution)
        for i in range(self.resolution):
            h1_i = self.robot.link_1.transform[:, :, i]
            transform = insertion_transform @ h1_i @ self.h2_mean
            
            cost_h1[i] = cost_total(
                insertion_point, insertion_transform, transform,
                point_cloud, pca_vector, pedicle_center,
                self.weight, volume, self.reach
            )
        
        # Find best first joint position
        i_min = np.argmin(cost_h1)
        h1_best = self.robot.link_1.transform[:, :, i_min]
        
        # Phase 2: Search in second joint space with best first joint
        cost_h2 = np.zeros(self.resolution)
        for i in range(self.resolution):
            h2_i = self.robot.link_2.transform[:, :, i]
            transform = insertion_transform @ h1_best @ h2_i
            
            cost_h2[i] = cost_total(
                insertion_point, insertion_transform, transform,
                point_cloud, pca_vector, pedicle_center,
                self.weight, volume, self.reach
            )
        
        # Find best second joint position
        i_min = np.argmin(cost_h2)
        h2_best = self.robot.link_2.transform[:, :, i_min]
        
        # Generate final trajectory
        best_transform = insertion_transform @ h1_best @ h2_best
        final_traj = gen_traj(insertion_transform, best_transform)
        
        return final_traj, np.min(cost_h2)
    
    def calculate_angles(self, trajectory):
        """
        Convert trajectory vector to vertical and horizontal angles
        """
        # Vertical angle (pitch) - in YZ plane
        traj_yz = np.array([0, trajectory[1], trajectory[2]])
        traj_yz_norm = np.linalg.norm(traj_yz)
        if traj_yz_norm > 0:
            traj_yz = traj_yz / traj_yz_norm
        
        vertical_angle = np.degrees(np.arctan2(
            np.linalg.norm(np.cross([0, 1, 0], traj_yz)),
            np.dot([0, 1, 0], traj_yz)
        ))
        
        # Adjust sign based on Z component
        if trajectory[2] < 0:
            vertical_angle = -vertical_angle
        
        # Horizontal angle (yaw) - in XY plane
        traj_xy = np.array([trajectory[0], trajectory[1], 0])
        traj_xy_norm = np.linalg.norm(traj_xy)
        if traj_xy_norm > 0:
            traj_xy = traj_xy / traj_xy_norm
        
        horizontal_angle = np.degrees(np.arctan2(
            np.linalg.norm(np.cross([0, 1, 0], traj_xy)),
            np.dot([0, 1, 0], traj_xy)
        ))
        
        # Adjust sign based on X component
        if trajectory[0] < 0:
            horizontal_angle = -horizontal_angle
        
        return vertical_angle, horizontal_angle