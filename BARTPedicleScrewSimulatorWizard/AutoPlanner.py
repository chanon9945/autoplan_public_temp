import numpy as np
from .RobotInit import Robot
from .CostFunctions import cost_total, gen_traj
import logging

class PedicleScrewAutoPlanner:
    """
    Automatic trajectory planning for pedicle screws
    """
    def __init__(self, resolution=1000, reach=100, weight=None, progress_callback=None):
        """
        Initialize the auto planner with the specified parameters.
        
        Parameters:
            resolution: Number of points to sample in the robot workspace
            reach: Maximum reach distance for trajectory planning
            weight: Weight factors for different cost components [distance, angle, boundary, density]
            progress_callback: Function to report progress (phase, iteration, max_iterations)
        """
        if weight is None:
            weight = [300, 1, 30, 0.05]
        
        self.resolution = resolution
        self.reach = reach
        self.weight = weight
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Initialize robot workspace
        self.logger.info(f"Initializing robot workspace with resolution={resolution}, reach={reach}")
        self.robot = Robot(resolution, reach)
        
        # Calculate mean transforms
        self.h1_mean, self.h2_mean = self.robot.get_mean_transforms()
    
    def plan_trajectory(self, vertebra, insertion_point):
        """
        Plan optimal trajectory for a given vertebra and insertion point
        
        Parameters:
            vertebra: Vertebra object containing anatomical information
            insertion_point: 3D coordinates [x, y, z] of the insertion point
            
        Returns:
            final_traj: Vector representing the optimal trajectory direction
            angles: Tuple of (vertical_angle, horizontal_angle) in degrees
            cost: Cost value of the final trajectory
        """
        self.logger.info(f"Planning trajectory for insertion point {insertion_point}")
        
        # Create insertion transform
        insertion_transform = np.eye(4)
        insertion_transform[0:3, 3] = insertion_point
        
        # Get pedicle center point
        if hasattr(vertebra, 'pedicle_point_cloud') and vertebra.pedicle_point_cloud:
            if hasattr(vertebra.pedicle_point_cloud, 'GetPoints'):
                points = vertebra.pedicle_point_cloud.GetPoints()
                if points and points.GetNumberOfPoints() > 0:
                    n_points = points.GetNumberOfPoints()
                    point_sum = np.zeros(3)
                    for i in range(n_points):
                        point = points.GetPoint(i)
                        point_sum += np.array(point)
                    pedicle_center = point_sum / n_points
                else:
                    pedicle_center = np.array(insertion_point)
            else:
                if hasattr(vertebra, 'pedicle_center_point'):
                    pedicle_center = vertebra.pedicle_center_point
                else:
                    pedicle_center = np.array(insertion_point)
        else:
            pedicle_center = np.array(insertion_point)
        
        self.logger.info(f"Pedicle center calculated at {pedicle_center}")
        
        # Run optimization - Phase 1: Search in first joint space
        self.logger.info("Phase 1: Searching in first joint space")
        cost_h1 = np.zeros(self.resolution)
        for i in range(self.resolution):
            h1_i = self.robot.link_1.transform[:, :, i]
            transform = insertion_transform @ h1_i @ self.h2_mean
            
            cost_h1[i] = cost_total(
                insertion_point, insertion_transform, transform,
                vertebra.point_cloud, vertebra.pcaVectors[:, 2], pedicle_center,
                self.weight, vertebra.maskedVolume, self.reach
            )
            
            # Report progress periodically
            if i % 10 == 0 and self.progress_callback:
                should_cancel = self.progress_callback(1, i, self.resolution)
                if should_cancel:
                    return None, (0, 0), float('inf')
        
        # Find best first joint position
        i_min = np.argmin(cost_h1)
        h1_best = self.robot.link_1.transform[:, :, i_min]
        self.logger.info(f"Phase 1 complete. Best joint index: {i_min}, Cost: {cost_h1[i_min]}")
        
        # Phase 2: Search in second joint space with best first joint
        self.logger.info("Phase 2: Searching in second joint space")
        cost_h2 = np.zeros(self.resolution)
        for i in range(self.resolution):
            h2_i = self.robot.link_2.transform[:, :, i]
            transform = insertion_transform @ h1_best @ h2_i
            
            cost_h2[i] = cost_total(
                insertion_point, insertion_transform, transform,
                vertebra.point_cloud, vertebra.pcaVectors[:, 2], pedicle_center,
                self.weight, vertebra.maskedVolume, self.reach
            )
            
            # Report progress periodically
            if i % 10 == 0 and self.progress_callback:
                should_cancel = self.progress_callback(2, i, self.resolution)
                if should_cancel:
                    return None, (0, 0), float('inf')
        
        # Find best second joint position
        i_min = np.argmin(cost_h2)
        h2_best = self.robot.link_2.transform[:, :, i_min]
        self.logger.info(f"Phase 2 complete. Best joint index: {i_min}, Cost: {cost_h2[i_min]}")
        
        # Generate final trajectory
        best_transform = insertion_transform @ h1_best @ h2_best
        final_traj = gen_traj(insertion_transform, best_transform)
        
        # Calculate angles for UI
        vertical_angle, horizontal_angle = self.calculate_angles(final_traj)
        self.logger.info(f"Final trajectory angles: vertical={vertical_angle}, horizontal={horizontal_angle}")
        
        return final_traj, (vertical_angle, horizontal_angle), np.min(cost_h2)
    
    def calculate_angles(self, trajectory):
        """
        Convert trajectory vector to vertical and horizontal angles
        
        Parameters:
            trajectory: 3D vector representing the trajectory direction
            
        Returns:
            vertical_angle: Pitch angle in degrees (up/down)
            horizontal_angle: Yaw angle in degrees (left/right)
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