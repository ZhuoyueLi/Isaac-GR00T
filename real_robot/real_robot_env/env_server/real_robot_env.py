from enum import Enum, auto
import cv2
import gymnasium as gym
import torch
from torchcontrol.transform import Rotation as R

from real_robot_env.robot.hardware_cameras import DiscreteCamera
from real_robot_env.robot.hardware_franka import ControlType, FrankaArm
from real_robot_env.robot.hardware_frankahand import FrankaHand
import numpy as np
from typing import Any, Dict, List, Optional


class ControlMode(Enum):
    END_EFFECTOR = auto()
    DELTA_END_EFFECTOR = auto()
    JOINTS = auto()

class RealRobotEnv(gym.Env):

    """

    This is a simple gym environment that forms an interface to the real lab environment.
    All needed cameras can be passed in the constructor.
    The used action and observation spaces are described in the variables self.observation_space and self.action_space.

    """

    def __init__(
        self,
        robot_arm: FrankaArm,
        robot_gripper: FrankaHand,
        control_mode: ControlMode = ControlMode.JOINTS,
        cameras: List[DiscreteCamera] = [],
    ):
        
        # Initialize robot
        self.control_mode = control_mode
        self.robot_arm, self.robot_gripper = self._setup_robot(
            robot_arm,
            robot_gripper,
        )

        # Connect to cameras
        self.cameras = cameras
        for cam in self.cameras:
            assert cam.connect(), f"Connection to {cam.name} failed"

        # Define observation space
        observation_space = {
            "joint_pos": gym.spaces.Box(
                shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
            ),
            "joint_vel": gym.spaces.Box(
                shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
            ),
            "ee_pos": gym.spaces.Box(
                shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
            ),
            "ee_vel": gym.spaces.Box(
                shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32
            ),
            "gripper_width": gym.spaces.Box(
                shape=(1,), low=-1, high=1, dtype=np.float64
            ),
        }
        for cam in self.cameras:
            observation_space[cam.name] = gym.spaces.Box(
                shape=(cam.height, cam.width, 3),  # Channels are RGB
                low=0,
                high=255,
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(observation_space)

        # Define action space
        if self.control_mode == ControlMode.JOINTS:
            self.action_space = gym.spaces.Box(
                shape=(8,),
                low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1]),
                high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]),
                dtype=np.float64
            )
        elif self.control_mode == ControlMode.END_EFFECTOR:
            # End effector position has the format: (xyz, orientation quat, grasp action)
            # Quaternions are in the same convention that torchcontrol uses: x, y, z, w
            self.action_space = gym.spaces.Box(
                shape=(8,),
                low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1]),
                high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]),
                dtype=np.float64
            )
        elif self.control_mode == ControlMode.DELTA_END_EFFECTOR:
            # Delta end effector postion has the format: (delta xyz, delta orientation quat, grasp action)
            # Quaternions are in the same convention that torchcontrol uses: x, y, z, w
            self.action_space = gym.spaces.Box(
                shape=(8,),
                low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1]),
                high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]),
                dtype=np.float64
            )
        else:
            raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        # Arm
        if self.control_mode == ControlMode.JOINTS:
            des_arm_joints = action[:7]
            # self.robot_arm.apply_commands(q_desired=des_arm_joints)
            self.robot_arm.go_to_within_limits(goal=des_arm_joints)

            
        elif self.control_mode == ControlMode.END_EFFECTOR:
            des_arm_pos = torch.from_numpy(action[:3])
            des_arm_quat = torch.from_numpy(action[3:7])
            self.robot_arm.apply_commands(ee_pos_desired=des_arm_pos, ee_quat_desired=des_arm_quat)
        elif self.control_mode == ControlMode.DELTA_END_EFFECTOR:
            current_ee = self.robot_arm.get_state().ee_pos
            des_delta_ee = torch.from_numpy(action[:7])
            des_arm_pos = current_ee[:3] + des_delta_ee[:3]
            des_arm_quat = R.functional.quaternion_multiply(des_delta_ee[3:7], current_ee[3:7])
            self.robot_arm.apply_commands(ee_pos_desired=des_arm_pos, ee_quat_desired=des_arm_quat)
            # TODO Currently, the policy DELTA_CARTESIAN_IMPEDANCE_CONTROL does not work properly
            # des_arm_delta_pos = torch.from_numpy(action[:3])
            # des_arm_delta_quat = torch.from_numpy(action[3:7])
            # self.robot_arm.apply_commands(ee_delta_pos_desired=des_arm_delta_pos, ee_delta_quat_desired=des_arm_delta_quat)
        else:
            raise NotImplementedError
        
        # Gripper
        # des_gripper_state = action[-1] # Grasps for < 0, else opens the gripper
        des_gripper_state = 1 if action[-1] >0 else -1 # Grasps for < 0, else opens the gripper
        self.robot_gripper.apply_commands(width=des_gripper_state)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, 0, False, False, info # TODO truncated can be helpful (time limit, robot constraint violation, ...)

    def reset(self) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:

        self.robot_arm.reset()
        self.robot_gripper.reset()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def close(self):

        self.robot_arm.close()
        self.robot_gripper.close()

        for cam in self.cameras:
            cam.close()

    def _get_obs(self) -> Dict:

        # Get robot state
        arm_state = self.robot_arm.get_state()
        gripper_width = self.robot_gripper.get_sensors()
        
        # Capture RGB images and remove depth
        imgs = [cam.get_sensors()["rgb"][:, :, :3] for cam in self.cameras]

        # Images should have the shape (height, width, 3) with values from [0,255] and the type np.uint8
        # Verify that images have the correct shape
        # TODO this resizing should not be required
        resized_imgs = []
        for img, cam in zip(imgs, self.cameras):
            if img.shape[0] != cam.height or img.shape[1] != cam.width:
                # TODO Maybe try cv.INTER_LANCZOS4 interpolation since OpenVLA uses lanczos3
                resized_imgs.append(cv2.resize(img, (cam.height, cam.width)))
            else:
                resized_imgs.append(img)
        processed_imgs = [img.astype(np.uint8) for img in resized_imgs]

        # Format observation
        obs = {
            "joint_pos": arm_state.joint_pos.cpu().numpy(),
            "joint_vel": arm_state.joint_vel.cpu().numpy(),
            "ee_pos": arm_state.ee_pos.cpu().numpy(),
            "ee_vel": arm_state.ee_vel.cpu().numpy(),
            # "gripper_width": gripper_width,
            # "gripper_width": -1 if gripper_width < 0.085/1.3 else 1,   # max_width =0.085
            "gripper_width": np.array([-1.0], dtype=np.float64) if gripper_width < 0.085/1.3 else np.array([1.0], dtype=np.float64)
        }
        for img, cam in zip(processed_imgs, self.cameras):
            obs[cam.name] = img

        return obs


    def _get_info(self) -> dict[str, Any]:
        return {}

    def _setup_robot(
        self,
        robot_arm,
        robot_gripper,
    ) -> tuple[FrankaArm, FrankaHand]:

        # Set Polymetis policy matching to the action space
        if self.control_mode == ControlMode.JOINTS:
            control_type = ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL
        elif self.control_mode == ControlMode.END_EFFECTOR:
            control_type = ControlType.CARTESIAN_IMPEDANCE_CONTROL
        elif self.control_mode == ControlMode.DELTA_END_EFFECTOR:
            control_type = ControlType.CARTESIAN_IMPEDANCE_CONTROL
            # TODO Currently, the policy DELTA_CARTESIAN_IMPEDANCE_CONTROL does not work properly
            #control_type = ControlType.DELTA_CARTESIAN_IMPEDANCE_CONTROL
        else:
            raise NotImplementedError
        robot_arm.control_type = control_type

        # Connect robot
        assert robot_arm.connect(), f"Connection to {robot_arm.name} failed"
        assert robot_gripper.connect(), f"Connection to {robot_gripper.name} failed"

        return robot_arm, robot_gripper
