import numpy as np
from .RobotInit import Robot
from .CostFunctions import cost_total, gen_traj
import logging
import concurrent.futures
import time

class PedicleScrewAutoPlanner:
    """
    Automatic trajectory planning for pedicle screws with parallel processing
    """
    def __init__(self, resolution=1000, reach=100, weight=None, n_jobs=4, progress_callback=None):
        if weight is None:
            weight = [300, 1, 30, 0.05]
        
        self.resolution = resolution
        self.reach = reach
        self.weight = weight
        # Ensure n_jobs is at least 1 and no more than 8
        self.n_jobs = max(1, min(n_jobs, 8))  
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Initialize robot workspace
        self.robot = Robot(resolution, reach)
        self.h1_mean, self.h2_mean = self.robot.get_mean_transforms()
    
    def _evaluate_cost(self, args):
        """Combined worker function for both joints"""
        joint_num, i, h1_transform, insertion_point, insertion_transform, pt_cloud, pc_vector, pedicle_center, volume, reach, weight = args
        
        if joint_num == 1:  # First joint
            transform = insertion_transform @ h1_transform @ self.h2_mean
        else:  # Second joint
            transform = insertion_transform @ h1_transform
        
        cost = cost_total(
            insertion_point, insertion_transform, transform,
            pt_cloud, pc_vector, pedicle_center,
            weight, volume, reach
        )
        return i, cost
    
    def plan_trajectory(self, vertebra, insertion_point):
        """Plan optimal trajectory using concurrent processing"""
        # Setup
        insertion_transform = np.eye(4)
        insertion_transform[0:3, 3] = insertion_point
        
        # Get pedicle center
        pedicle_center = np.array(insertion_point)
        if hasattr(vertebra, 'pedicle_center_point'):
            pedicle_center = vertebra.pedicle_center_point
        
        # Phase 1: First joint search
        self.logger.info("Phase 1: Searching in first joint space")
        cost_h1 = np.zeros(self.resolution)
        
        # Process in smaller chunks
        chunk_size = min(20, self.resolution // 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for start_idx in range(0, self.resolution, chunk_size):
                end_idx = min(start_idx + chunk_size, self.resolution)
                
                # Create task arguments
                tasks = []
                for i in range(start_idx, end_idx):
                    h1_i = self.robot.link_1.transform[:, :, i]
                    tasks.append((
                        1, i, h1_i, insertion_point, insertion_transform,
                        vertebra.point_cloud, vertebra.pcaVectors[:, 2], pedicle_center,
                        vertebra.maskedVolume, self.reach, self.weight
                    ))
                
                # Submit tasks
                futures = [executor.submit(self._evaluate_cost, task) for task in tasks]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, cost = future.result()
                        cost_h1[i] = cost
                    except Exception as e:
                        self.logger.error(f"Error in worker: {str(e)}")
                
                # Report progress and check for cancellation
                if self.progress_callback:
                    should_cancel = self.progress_callback(1, end_idx, self.resolution)
                    if should_cancel:
                        return None, (0, 0), float('inf')
        
        # Find best first joint position
        i_min = np.argmin(cost_h1)
        h1_best = self.robot.link_1.transform[:, :, i_min]
        
        # Phase 2: Second joint search
        self.logger.info("Phase 2: Searching in second joint space")
        cost_h2 = np.zeros(self.resolution)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for start_idx in range(0, self.resolution, chunk_size):
                end_idx = min(start_idx + chunk_size, self.resolution)
                
                # Create task arguments
                tasks = []
                for i in range(start_idx, end_idx):
                    h2_i = self.robot.link_2.transform[:, :, i]
                    combined = insertion_transform @ h1_best @ h2_i
                    tasks.append((
                        2, i, combined, insertion_point, insertion_transform,
                        vertebra.point_cloud, vertebra.pcaVectors[:, 2], pedicle_center,
                        vertebra.maskedVolume, self.reach, self.weight
                    ))
                
                # Submit tasks
                futures = [executor.submit(self._evaluate_cost, task) for task in tasks]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, cost = future.result()
                        cost_h2[i] = cost
                    except Exception as e:
                        self.logger.error(f"Error in worker: {str(e)}")
                
                # Report progress and check for cancellation
                if self.progress_callback:
                    should_cancel = self.progress_callback(2, end_idx, self.resolution)
                    if should_cancel:
                        return None, (0, 0), float('inf')
        
        # Find best position and generate trajectory
        i_min = np.argmin(cost_h2)
        h2_best = self.robot.link_2.transform[:, :, i_min]
        best_transform = insertion_transform @ h1_best @ h2_best
        final_traj = gen_traj(insertion_transform, best_transform)
        
        # Calculate angles for UI
        vertical_angle, horizontal_angle = self.calculate_angles(final_traj)
        
        return final_traj, (vertical_angle, horizontal_angle), np.min(cost_h2)
    
    def calculate_angles(self, trajectory):
        """Convert trajectory vector to vertical and horizontal angles"""
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