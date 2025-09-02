from omegaconf import DictConfig
from real_robot_env.robot.hardware_devices import DiscreteDevice, ContinuousDevice
from datetime import datetime
import torch
from pathlib import Path
import cv2
import shutil
from enum import Enum, auto
from typing import NamedTuple, Optional
import time
import hydra
from hydra.utils import instantiate

from real_robot_env.robot.hardware_robot import RobotArm, RobotHand
from utils.keyboard_input import NonBlockingKeyPress

class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()


class Robot:
    def __init__(
        self,
        name: str,
        arm: RobotArm,
        gripper: RobotHand,
    ):
        self.robot_arm = arm
        self.robot_gripper = gripper
        self.name = name
        self.robot_arm.name = f"{name} arm"
        self.robot_gripper.name = f"{name} gripper"

        self.is_connected = False

    def connect(self):
        assert self.robot_arm.connect(), f"Connection to {self.robot_arm.name} failed"
        assert self.robot_gripper.connect(), f"Connection to {self.robot_gripper.name} failed"
        self.is_connected = True

    def close(self):
        if not self.is_connected:
            return print("Robot is already not connected")

        self.robot_arm.close()
        self.robot_gripper.close()

        self.is_connected = False

    def reset(self):
        self.robot_arm.reset()
        self.robot_gripper.reset()


class TeleoperationPair(NamedTuple):
    leader_robot: Robot
    follower_robot: Robot


class CollectionData:
    def __init__(self):
        self.joint_pos_list = []
        self.joint_vel_list = []
        self.ee_pos_list = []
        self.ee_vel_list = []
        self.gripper_state_list = []

    def append(self, joint_pos: Optional[torch.Tensor], joint_vel: Optional[torch.Tensor], ee_pos: Optional[torch.Tensor], ee_vel: Optional[torch.Tensor], gripper_state: int):
        self.joint_pos_list.append(joint_pos)
        self.joint_vel_list.append(joint_vel)
        self.ee_pos_list.append(ee_pos)
        self.ee_vel_list.append(ee_vel)
        self.gripper_state_list.append(gripper_state)

    def save(self, path: Path):

        tensor_lists = [self.joint_pos_list, self.joint_vel_list, self.ee_pos_list, self.ee_vel_list]
        paths = [path / "joint_pos.pt", path / "joint_vel.pt", path / "ee_pos.pt", path / "ee_vel.pt"]

        for d, p in zip(tensor_lists, paths):

            if len(d) == 0 or None in d:  # E.g. Gello only provides joint_pos and returns None for the rest
                print(f"Skip saving '{p}' since it is empty")
                continue

            d_stacked = torch.stack(d)
            torch.save(d_stacked, p)
            print(f"Successfully saved '{p}'")

        gripper_state_list = torch.Tensor(self.gripper_state_list)
        gripper_state_path = path / "gripper_state.pt"
        torch.save(gripper_state_list, gripper_state_path)
        print(f"Successfully saved '{gripper_state_path}'")


class DataCollectionManager:
    """
    This class handles the core logic of collecting data.
    It also includes an interactive interface to communicate with the user.

    Args:
        teleoperation_pairs: A list with leader follower pairs where each Robot can e.g. be a Franka Panda or Gello.
        data_dir: Directory to save the data.
        teleoperation_type: Whether the teleoperation should be in joint space or task space. 
                            Note that the FrankaArm needs to run a corresponding policy.
        discrete_devices: List of devices (f.e. cameras), that capture environment in frames.
        continuous_devices: List of devices (f.e. cameras/microphone), that capture environment in a single recording (video).
        capture_interval: Sets the frequency at which data is collected
        initial_robot_sync: Allows to initially synchronize leader and follower, which is needed e.g. for Gello as leader.
    """
        
    def __init__(
        self,
        teleoperation_pairs: list[TeleoperationPair],
        data_dir: Path,
        teleoperation_type: TeleoperationType,
        discrete_devices: list[DiscreteDevice] = [],
        continuous_devices: list[ContinuousDevice] = [],
        capture_interval: int = 0,
        initial_robot_sync: bool  = False,
    ):
        self.teleoperation_type = teleoperation_type
        self.teleoperation_pairs = teleoperation_pairs

        for teleoperation_pair in self.teleoperation_pairs:
            teleoperation_pair.leader_robot.connect()
            teleoperation_pair.follower_robot.connect()

        self.discrete_devices = discrete_devices
        self.continuous_devices = continuous_devices
        self.__setup_devices()
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        self.timestamps = []
        self.cur_timestep = 0
        self.capture_interval = capture_interval
        self.initial_robot_sync = initial_robot_sync

    def start_key_listener(self):
        print("📦 Press 'n' to collect new data or 'q' to quit data collection")

        with NonBlockingKeyPress() as kp:
            quit = False
            while not quit:

                # Update input
                key = kp.get_data()
                if key == "q":
                    quit = True
                elif key == "n":
                    quit = False

                if key == "n":
                    print("📯 Preparing for new data collection")

                    self.__create_new_recording_dir()
                    self.__create_empty_data()
                    self.__reset_robots()
                    self.__start_continuous_recordings()

                    print("🚀 Start! Press 's' to save collected data or 'd' to discard.")

                    if self.initial_robot_sync:
                        self.__sync_robots()

                    self.timestamps = []
                    self.cur_timestep = 0
                    collect = True
                    while collect:

                        # Update input
                        key = kp.get_data()
                        if key in ["s", "d"]:
                            collect = False

                        self.__collection_step()

                    else:
                        self.__stop_continuous_recordings()
                        if key == "s":
                            print("Saving data ...")

                            self.__save_data()

                            print("Saved!")
                        elif key == "d":
                            print("Discarding data ...")

                            shutil.rmtree(self.record_dir)
                            self.__discard_continuous_recordings()

                            print("Discarded!")

                        print(
                            "📦 Press 'n' to collect new data or 'q' to quit data collection"
                        )

        print("⌛ Ending data collection...")
        self.__close_hardware_connections()

    def __setup_devices(self):
        for device in self.discrete_devices:
            assert device.connect(), f"Connection to {device.name} failed"
        for device in self.continuous_devices:
            assert device.connect(), f"Connection to {device.name} failed"

    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        for teleoperation_pair in self.teleoperation_pairs:
            self.leader_robot_dir = (
                self.record_dir / teleoperation_pair.leader_robot.name
            )
            self.leader_robot_dir.mkdir()

            self.follower_robot_dir = (
                self.record_dir / teleoperation_pair.follower_robot.name
            )
            self.follower_robot_dir.mkdir()

        self.sensors_dir = self.record_dir / "sensors"
        self.sensors_dir.mkdir()

        for device in self.discrete_devices:
            device_dir = self.sensors_dir / device.name
            device_dir.mkdir()

        for device in self.continuous_devices:
            device_dir = self.sensors_dir / device.name
            device_dir.mkdir()

    def __create_empty_data(self):
        self.robot_to_data: dict[Robot, CollectionData] = {}

        for teleoperation_pair in self.teleoperation_pairs:
            self.robot_to_data[teleoperation_pair.leader_robot] = CollectionData()
            self.robot_to_data[teleoperation_pair.follower_robot] = CollectionData()

    def __reset_robots(self):
        for teleoperation_pair in self.teleoperation_pairs:
            teleoperation_pair.leader_robot.reset()
            teleoperation_pair.follower_robot.reset()

    def __sync_robots(self):
        for teleoperation_pair in self.teleoperation_pairs:
            # Move slowly to the leader's position
            leader_arm_state = teleoperation_pair.leader_robot.robot_arm.get_state()
            teleoperation_pair.follower_robot.robot_arm.go_to_within_limits(leader_arm_state.joint_pos, max_vel_norm_factor=0.5)

    def __start_continuous_recordings(self):
        for device in self.continuous_devices:
            device.start_recording()

    def __stop_continuous_recordings(self):
        for device in self.continuous_devices:
            device.stop_recording()

    def __discard_continuous_recordings(self):
        for device in self.continuous_devices:
            device.delete_recording()

    def __collection_step(self):
        for teleoperation_pair in self.teleoperation_pairs:
            self.leader_arm = teleoperation_pair.leader_robot.robot_arm
            self.leader_gripper = teleoperation_pair.leader_robot.robot_gripper
            self.follower_arm = teleoperation_pair.follower_robot.robot_arm
            self.follower_gripper = teleoperation_pair.follower_robot.robot_gripper

            leader_arm_state = self.leader_arm.get_state()
            leader_gripper_width = self.leader_gripper.get_sensors().item()
            leader_gripper_state = self.__get_follower_gripper_state(
                leader_gripper_width, self.leader_gripper.max_width / 2
            )

            if self.teleoperation_type is TeleoperationType.JOINT_SPACE:
                self.follower_arm.apply_commands(
                    q_desired=leader_arm_state.joint_pos,
                    qd_desired=leader_arm_state.joint_vel,
                )
            elif self.teleoperation_type is TeleoperationType.TASK_SPACE:
                self.follower_arm.apply_commands(
                    ee_pos_desired=leader_arm_state.ee_pos[:3],
                    ee_quat_desired=leader_arm_state.ee_pos[3:],
                    ee_vel_desired=leader_arm_state.ee_vel[:3],
                    ee_rvel_desired=leader_arm_state.ee_vel[3:],
                )
            else:
                raise ValueError("The given teleoperation type is invalid")

            self.follower_gripper.apply_commands(leader_gripper_state)

            follower_arm_state = self.follower_arm.get_state()
            follower_gripper_width = self.follower_gripper.get_sensors().item()
            follower_gripper_state = self.__get_follower_gripper_state(
                follower_gripper_width, self.follower_gripper.max_width / 2
            )

            cur_time = time.time()  # Store timestamps as seconds since the Unix epoch

            if not self.timestamps or cur_time - self.timestamps[-1] >= self.capture_interval:
                self.robot_to_data[teleoperation_pair.leader_robot].append(
                    leader_arm_state.joint_pos,
                    leader_arm_state.joint_vel,
                    leader_arm_state.ee_pos,
                    leader_arm_state.ee_vel,
                    leader_gripper_state
                )
                self.robot_to_data[teleoperation_pair.follower_robot].append(
                    follower_arm_state.joint_pos,
                    follower_arm_state.joint_vel,
                    follower_arm_state.ee_pos,
                    follower_arm_state.ee_vel,
                    follower_gripper_state
                )

                self.timestamps.append(cur_time)

                for device in self.discrete_devices:
                    device.store_last_frame(self.sensors_dir / device.name, f"{self.cur_timestep + device.start_frame_latency}")
            
                self.cur_timestep += 1
            #else: print(f"too quick: {self.cur_timestep}")

    def __get_follower_gripper_state(self, leader_gripper_width: float, thresh: float):
        if leader_gripper_width < thresh:
            return -1
        else:
            return 1

    def __save_data(self):

        timestamps_path = self.record_dir / "timestamps.pt"
        torch.save(torch.tensor(self.timestamps, dtype=torch.float64), timestamps_path)
        print(f"Successfully saved '{timestamps_path}'")
        for teleoperation_pair in self.teleoperation_pairs:
            leader_data = self.robot_to_data[teleoperation_pair.leader_robot]
            follower_data = self.robot_to_data[teleoperation_pair.follower_robot]
            leader_data.save(self.record_dir / teleoperation_pair.leader_robot.name)
            follower_data.save(self.record_dir / teleoperation_pair.follower_robot.name)
      
        for device in self.continuous_devices:
            device.store_recording(self.sensors_dir / device.name, "recording", self.timestamps)
        
        # determine average frame rate from timestamps
        print(f"Robot states frame rate: {len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0]):.2f} Hz")
        
        for device in self.discrete_devices:
            device.timestamps = []

    def __close_hardware_connections(self):
        for teleoperation_pair in self.teleoperation_pairs:
            teleoperation_pair.leader_robot.close()
            teleoperation_pair.follower_robot.close()

        for device in self.discrete_devices:
            device.close()

        for device in self.continuous_devices:
            device.close()



@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig):

    # Instantiate objects from config
    teleoperation_pairs = [instantiate(p) for p in cfg.teleoperation_pairs.values()]
    
    # Check if continuous devices are defined in the config
    if 'continuous_devices' in cfg:
        continuous_devices = [instantiate(d) for d in cfg.continuous_devices.values()]
    else:
        continuous_devices = []
    # Check if discrete devices are defined in the config
    if 'discrete_devices' in cfg:
        discrete_devices = [instantiate(d) for d in cfg.discrete_devices.values()]
    else:
        discrete_devices = []

    # Start data collection
    data_collection_manager = DataCollectionManager(
        teleoperation_pairs=teleoperation_pairs,
        data_dir=Path(cfg.data_dir),
        teleoperation_type=TeleoperationType[cfg.teleoperation_type],
        continuous_devices=continuous_devices,
        discrete_devices=discrete_devices,
        capture_interval=cfg.capture_interval,
        initial_robot_sync=cfg.initial_robot_sync,
    )
    data_collection_manager.start_key_listener()

if __name__ == "__main__":
    main()