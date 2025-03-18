import numpy as np
from .RobotInit import Robot
from .CostFunctions import cost_total, gen_traj
import logging

class PedicleScrewAutoPlanner:
    """
    Automatic trajectory planning for pedicle screws using robot kinematics
    and cost function optimization.
    """
    def __init__(self, resolution=1000, reach=100, weight=None, progress_callback=None):
        """
        Initialize the auto planner with the specified parameters.
        
        Parameters:
            resolution: Number of points to sample in the robot workspace
            reach: Maximum reach distance for trajectory planning
            weight: Weight factors for cost components [distance, angle, boundary, density]
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
        
        # Calculate mean transforms for the robot links
        self.h1_mean, self.h2_mean = self.robot.get_mean_transforms()
    
    def plan_trajectory(self, vertebra, insertion_point):
        """
        Plan optimal trajectory for a given vertebra and insertion point.
        
        Parameters:
            vertebra: Vertebra object containing anatomical information
            insertion_point: 3D coordinates [x, y, z] of the insertion point
            
        Returns:
            final_traj: Vector representing the optimal trajectory direction
            angles: Tuple of (vertical_angle, horizontal_angle) in degrees
            cost: Cost value of the final trajectory
        """
        self.logger.info(f"Planning trajectory for insertion point {insertion_point}")
        
        # Validate inputs
        if vertebra is None:
            self.logger.error("Vertebra object is None")
            return np.array([0, 1, 0]), (0, 0), float('inf')
        
        if insertion_point is None or np.any(np.isnan(insertion_point)):
            self.logger.error("Invalid insertion point")
            return np.array([0, 1, 0]), (0, 0), float('inf')
        
        # Create insertion transform (4x4 homogeneous transform matrix)
        insertion_transform = np.eye(4)
        insertion_transform[0:3, 3] = insertion_point
        
        # Get pedicle center point (fallback to insertion point if not available)
        pedicle_center = None
        if hasattr(vertebra, 'pedicle_center_point') and vertebra.pedicle_center_point is not None:
            pedicle_center = vertebra.pedicle_center_point
        elif hasattr(vertebra, 'centroid') and vertebra.centroid is not None:
            pedicle_center = vertebra.centroid
        else:
            pedicle_center = np.array(insertion_point)
            self.logger.warning("Using insertion point as pedicle center (no center point found)")
        
        self.logger.info(f"Pedicle center: {pedicle_center}")
        
        # Get pedicle axis (PCA vector)
        pedicle_axis = None
        if hasattr(vertebra, 'pcaVectors') and vertebra.pcaVectors is not None:
            pedicle_axis = vertebra.pcaVectors[:, 2]  # Use 3rd principal component
        else:
            # Default to vertical axis if PCA not available
            pedicle_axis = np.array([0, 1, 0])
            self.logger.warning("Using default vertical axis (no PCA vectors found)")
        
        # Get point cloud for collision detection
        if not hasattr(vertebra, 'point_cloud') or vertebra.point_cloud is None:
            self.logger.warning("No point cloud available for collision detection")
            # Create an empty point cloud as fallback
            import vtk
            empty_points = vtk.vtkPoints()
            empty_polydata = vtk.vtkPolyData()
            empty_polydata.SetPoints(empty_points)
            vertebra.point_cloud = empty_polydata
        
        # Get volume for density calculation
        if not hasattr(vertebra, 'maskedVolume') or vertebra.maskedVolume is None:
            self.logger.warning("No masked volume available for density calculation")
        
        # Phase 1: Search in first joint space
        self.logger.info("Phase 1: Searching in first joint space")
        cost_h1 = np.full(self.resolution, float('inf'))
        
        for i in range(self.resolution):
            # Report progress
            if self.progress_callback and i % 5 == 0:
                # Use a safer approach to call progress_callback
                try:
                    should_cancel = self.progress_callback(1, i, self.resolution)
                    # should_cancel might be returned in different ways depending on Qt version
                    if should_cancel:
                        self.logger.info("Optimization canceled by user")
                        return np.array([0, 1, 0]), (0, 0), float('inf')
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {str(e)}")
                    # Continue even if the callback has an error
            
            # Get transform for this joint position
            h1_i = self.robot.link_1.transform[:, :, i]
            transform = insertion_transform @ h1_i @ self.h2_mean
            
            # Calculate trajectory direction
            traj = gen_traj(insertion_transform, transform)
            
            try:
                # Calculate cost for this trajectory
                total_cost, _ = cost_total(
                    insertion_point, 
                    traj,
                    vertebra.point_cloud, 
                    pedicle_axis, 
                    pedicle_center,
                    self.weight, 
                    vertebra.maskedVolume, 
                    self.reach
                )
                cost_h1[i] = total_cost
            except Exception as e:
                self.logger.error(f"Error calculating cost for joint 1, position {i}: {str(e)}")
                cost_h1[i] = float('inf')
        
        # Find best first joint position
        min_cost_idx = np.argmin(cost_h1)
        min_cost = cost_h1[min_cost_idx]
        
        # Check if valid solution was found
        if np.isinf(min_cost):
            self.logger.warning("No valid solution found in phase 1")
            return np.array([0, 1, 0]), (0, 0), float('inf')
        
        h1_best = self.robot.link_1.transform[:, :, min_cost_idx]
        joint1_angle = self.robot.link_1.joint_limit[min_cost_idx]
        
        self.logger.info(f"Phase 1 complete. Best joint index: {min_cost_idx}, Cost: {min_cost}, Angle: {joint1_angle}")
        
        # Phase 2: Search in second joint space with fixed first joint
        self.logger.info("Phase 2: Searching in second joint space")
        cost_h2 = np.full(self.resolution, float('inf'))
        
        for i in range(self.resolution):
            # Report progress
            if self.progress_callback and i % 5 == 0:
                # Use a safer approach to call progress_callback
                try:
                    should_cancel = self.progress_callback(2, i, self.resolution)
                    # should_cancel might be returned in different ways depending on Qt version
                    if should_cancel:
                        self.logger.info("Optimization canceled by user")
                        return np.array([0, 1, 0]), (0, 0), float('inf')
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {str(e)}")
                    # Continue even if the callback has an error
            
            # Get transform for this joint position
            h2_i = self.robot.link_2.transform[:, :, i]
            transform = insertion_transform @ h1_best @ h2_i
            
            # Calculate trajectory direction
            traj = gen_traj(insertion_transform, transform)
            
            try:
                # Calculate cost for this trajectory
                total_cost, _ = cost_total(
                    insertion_point, 
                    traj,
                    vertebra.point_cloud, 
                    pedicle_axis, 
                    pedicle_center,
                    self.weight, 
                    vertebra.maskedVolume, 
                    self.reach
                )
                cost_h2[i] = total_cost
            except Exception as e:
                self.logger.error(f"Error calculating cost for joint 2, position {i}: {str(e)}")
                cost_h2[i] = float('inf')
        
        # Find best second joint position
        min_cost_idx = np.argmin(cost_h2)
        min_cost = cost_h2[min_cost_idx]
        
        # Check if valid solution was found
        if np.isinf(min_cost):
            self.logger.warning("No valid solution found in phase 2")
            return np.array([0, 1, 0]), (0, 0), float('inf')
        
        h2_best = self.robot.link_2.transform[:, :, min_cost_idx]
        joint2_angle = self.robot.link_2.joint_limit[min_cost_idx]
        
        self.logger.info(f"Phase 2 complete. Best joint index: {min_cost_idx}, Cost: {min_cost}, Angle: {joint2_angle}")
        
        # Generate final trajectory
        best_transform = insertion_transform @ h1_best @ h2_best
        final_traj = gen_traj(insertion_transform, best_transform)
        
        # Calculate angles for UI
        vertical_angle, horizontal_angle = self.calculate_angles(final_traj)
        self.logger.info(f"Final trajectory angles: vertical={vertical_angle}, horizontal={horizontal_angle}")
        
        return final_traj, (vertical_angle, horizontal_angle), min_cost
    
    def calculate_angles(self, trajectory):
        """
        Convert trajectory vector to vertical and horizontal angles.
        
        Parameters:
            trajectory: 3D vector representing the trajectory direction
            
        Returns:
            vertical_angle: Pitch angle in degrees (up/down)
            horizontal_angle: Yaw angle in degrees (left/right)
        """
        try:
            # Normalize trajectory
            trajectory = trajectory / np.linalg.norm(trajectory)
            
            # Vertical angle (pitch) - in YZ plane
            # Project to YZ plane first
            traj_yz = np.array([0, trajectory[1], trajectory[2]])
            traj_yz_norm = np.linalg.norm(traj_yz)
            
            if traj_yz_norm > 1e-6:
                traj_yz = traj_yz / traj_yz_norm
                # Calculate angle with Y axis [0,1,0]
                cos_angle = np.clip(np.dot(traj_yz, [0, 1, 0]), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                # Determine sign based on Z component
                vertical_angle = np.degrees(angle) * (1 if trajectory[2] < 0 else -1)
            else:
                vertical_angle = 0.0
            
            # Horizontal angle (yaw) - in XY plane
            # Project to XY plane first
            traj_xy = np.array([trajectory[0], trajectory[1], 0])
            traj_xy_norm = np.linalg.norm(traj_xy)
            
            if traj_xy_norm > 1e-6:
                traj_xy = traj_xy / traj_xy_norm
                # Calculate angle with Y axis [0,1,0]
                cos_angle = np.clip(np.dot(traj_xy, [0, 1, 0]), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                # Determine sign based on X component
                horizontal_angle = np.degrees(angle) * (1 if trajectory[0] > 0 else -1)
            else:
                horizontal_angle = 0.0
            
            return vertical_angle, horizontal_angle
            
        except Exception as e:
            self.logger.error(f"Error calculating angles: {str(e)}")
            return 0.0, 0.0