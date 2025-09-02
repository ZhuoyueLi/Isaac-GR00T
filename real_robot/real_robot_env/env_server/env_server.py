import itertools
import threading
from typing import Dict, Generic, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import zmq


T = TypeVar("T", bound=gym.Env)
class EnvServer(Generic[T]):

    """
    
    This class creates a gym environment of the given type and allows accessing it through a ZMQ server.
    Apart from the gym functions reset() and step(), a function get_observation() is also provided to get a new observation.
    The server uses tcp and runs per default on port 6060.
    An example client is described in the class EnvClient.

    Example Usage:

        from real_robot_env.env_server import EnvServer
        from real_robot_env.real_robot_env import RealRobotEnv, ControlMode
        from real_robot_env.robot.hardware_depthai import DepthAI

        # Initialize cameras
        discrete_cameras = [
            DepthAI(
                device_id='1844301021D9BF1200',
                name='primary_camera',
                height=256,
                width=256
            ),
            DepthAI(
                device_id='1844301071E7AB1200',
                name='secondary_camera',
                height=256,
                width=256
            )
        ]

        # Create env
        env = RealRobotEnv(
            robot_name = "Test arm",
            robot_ip_address = "10.10.10.110",
            robot_arm_port = 50051,
            robot_gripper_port = 50052,
            control_mode = ControlMode.JOINTS,
            reset_pose = [0.2719, -0.5165, 0.2650, -1.6160, -0.0920, 1.6146, -1.9760],
            cameras = discrete_cameras,
        )

        # Create env server
        server = EnvServer[RealRobotEnv](
            env = env,
            host = "127.0.0.1",
            port = 6060,
        )

        # Start server
        server.serve()

    """

    def __init__(
        self,
        env: T,
        host: str = "127.0.0.1",
        port: int = 6060,
    ):
        super().__init__()

        # Gym environment
        self.env = env

        # Configure ZMQ server
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)

        # Server messages
        self._timout_message = f"Timeout when waiting for request"
        self._running_message = f"Successfully running Gym Environment Server"
        self._spinner = itertools.cycle(["-", "\\", "|", "/"])
        
        # Events
        self._stop_event = threading.Event()

    def serve(self) -> None:
        
        """
        Start the server and block until a stop is signaled.
        """

        # Set timeout to 1000 ms
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  

        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv_json()

                # Print server status
                self._print_running()

                # Call the corresponding function
                command = message["command"]
                if command == "reset":
                    obs, info = self.env.reset()
                    results = {
                        "observation": obs,
                        "info": info,
                    }
                elif command == "step":
                    action = self._receive_action(message)
                    obs, reward, term, trunc, info = self.env.step(action)
                    results = {
                        "observation": obs,
                        "reward": reward,
                        "terminated": term,
                        "truncated": trunc,
                        "info": info,
                    }
                elif command == "get_observation":
                    # This method is not part of the gym interface and might not be supported by self.env
                    if not hasattr(self.env, "_get_obs"):
                        raise NotImplementedError(
                            f"The underlying gym environment does not support the command: {command}"
                        )
                    obs = self.env._get_obs()
                    results = {
                        "observation": obs,
                    }
                else:
                    raise NotImplementedError(
                        f"Invalid command: {command}"
                    )

                # Send response
                self._send_results(results)

            except zmq.Again:
                # Handle timeout and try again
                self._print_timout()

    def stop(self) -> None:
        
        """
        Signal that the server should stop.
        """

        self._stop_event.set()

    def _receive_action(self, message: Dict[str, Union[str, np.ndarray]]):

        flags = 0

        # Get metadata
        metadata = message["action_metadata"]

        # Receive action
        data = self._socket.recv(flags, copy=False, track=False)

        # Reconstruct action
        buf = memoryview(data)
        action = np.frombuffer(buf, dtype=metadata["dtype"]).reshape(metadata["shape"])

        return action

    def _send_results(self, results: Dict[str, Union[str, int, float, bool, np.ndarray, Dict[str, str]]]):

        flags = 0

        # Determine observation metadata
        obs = results.pop("observation")
        obs_metadata = {
            key: {
                "dtype": str(value.dtype),
                "shape": value.shape,
            }
            for key, value in obs.items()
        }
        results["observation_metadata"] = obs_metadata

        # Send metadata and simple results
        self._socket.send_json(results, flags | zmq.SNDMORE)

        # Send observation data
        data_parts = list(obs.values())
        self._socket.send_multipart(data_parts, flags, copy=False, track=False)

    def _print_timout(self) -> None:
        print(f"\r{self._timout_message} [{next(self._spinner)}]{' '*20}", end="", flush=True)

    def _print_running(self) -> None:
        print(f"\r{self._running_message}{' '*20}", end="", flush=True)
