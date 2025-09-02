import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import List
from real_robot_env.env_server.env_server import EnvServer
from real_robot_env.env_server.real_robot_env import RealRobotEnv, ControlMode
from real_robot_env.robot.hardware_cameras import DiscreteCamera
from real_robot_env.robot.hardware_robot import RobotArm, RobotHand


def launch_server(
    robot_arm: RobotArm,
    robot_hand: RobotHand,
    discrete_cameras: List[DiscreteCamera],
    control_mode: ControlMode,
    host: str = "127.0.0.1",
    port: int = 6060,

):

    # Create env
    env = RealRobotEnv(
        robot_arm,
        robot_hand,
        control_mode = control_mode,
        cameras = discrete_cameras,
    )
    
    # Create env server
    server = EnvServer[RealRobotEnv](
        env = env,
        host = host,
        port = port,
    )
    
    # Start server
    server.serve()


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: DictConfig):

    # Instantiate objects from config
    discrete_cameras = [instantiate(d) for d in cfg.discrete_devices.values()]
    robot_arm = instantiate(cfg.robot_arm)
    robot_hand = instantiate(cfg.robot_hand)


    # Start server
    launch_server(
        robot_arm,
        robot_hand,
        discrete_cameras,
        ControlMode[cfg.control_mode],
        cfg.host,
        cfg.port
    )


if __name__ == "__main__":

    main()