from typing import Dict, Tuple, Union
import cv2
import gymnasium as gym
import numpy as np
import zmq
import time

class EnvClient(gym.Env):
    # rr_env is THE standard, this interface exists as a means to adapt obs/actions to/from rr_envs format
    def to_standardized_obs(self, obs):
        return obs
    
    def from_standardized_action(self, action):
        return action
    
    # .. fill the rest of abstract interface when i have time


class RREnvClient(EnvClient):

    """
    
    This class is a gym environment that communicates with an EnvServer to implement its methods.
    The outputs are adapted to work with Octo.

    Example Usage:

        from real_robot_env.env_client import EnvClient
        import numpy as np

        client = EnvClient(
            host = "127.0.0.1",
            port = 6060,
        )

        assert client.connect()

        obs, info = client.reset()
        obs, reward, term, trunc, info = client.step(np.array([0.2719, -0.5165, 0.2650, -1.6160, -0.0920, 1.6146, -1.7760, -1]))

        client.close()
    
    Launch Server with:

        TODO

    """

    def __init__(
            self,
            name = "Environment Client",
            host: str = "127.0.0.1",
            port: int = 6060,
        ):

        self.name = name
        self.host = host
        self.port = port
        self._addr = f"tcp://{self.host}:{self.port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

        self.observation_space = None
        # This can be either the joint values or (delta xyz, delta quaternion as xyzw, grasp action)
        self.action_space = gym.spaces.Box(
            low = np.zeros((7,)),
            high = np.ones((7,)),   
            dtype = np.float64,
        )

        self.prev_time = time.time()
        self.hz = 20 # dictates step duration

    def connect(self) -> bool:

        # Connect to server
        print(f"Connecting to {self.name}...")
        try:
            self._socket.connect(self._addr)
            print("Success")
        except Exception as e:
            print("Failed with exception: ", e)
            return False

        # Update observation space with actual image shapes
        reference_obs = self.get_observation()
        self.observation_space = {
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
            "primary_camera": gym.spaces.Box(
                low = np.zeros(reference_obs["top_cam"].shape),
                high = 255 * np.ones(reference_obs["top_cam"].shape),
                dtype = np.uint8,
            ),
            "secondary_camera": gym.spaces.Box(
                low = np.zeros(reference_obs["side_cam"].shape),
                high = 255 * np.ones(reference_obs["side_cam"].shape),
                dtype = np.uint8,
            ),
        }
        # import matplotlib.pyplot as plt
        # plt.imshow(reference_obs["top_cam"])
        # plt.axis("off")
        # plt.show()
        return True

    def close(self) -> bool:

        self._socket.disconnect(self._addr)
        print(f"Closed connection to {self.name}")
        return True
    
    def handle_freq(self):
        # TODO: FIXME sync hardware time
        curr_time = time.time()
        elapsed = max(curr_time - self.prev_time, 0)
        self.prev_time = curr_time

        duration = 1 / self.hz
        remaining = max(duration - elapsed, 0)
        if remaining == 0:
            print("Step took %0.4fs, step duration: %0.4fs"%(elapsed, duration))
        else:
            time.sleep(remaining)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, str]]:
            
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        self.handle_freq()
        
        # Convert action
        action = self._convert_action(action)

        # Send request
        self._send_step_request(action)  # TODO Add parameter for blocking
        
        # Receive response
        results = self._receive_results()

        # Convert observation
        obs = self._convert_obs(results["observation"])

        return obs, results["reward"], results["terminated"], results["truncated"], results["info"]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_reset_request()
        
        # Receive response
        results = self._receive_results()

        # Convert observation
        obs = self._convert_obs(results["observation"])

        return obs, results["info"]
    
    def get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_get_observation_request()
   
        # Receive response
        results = self._receive_results()
        # Convert observations
        obs = self._convert_obs(results["observation"])
        return obs

    def _convert_action(self, action):

        return action  # Conversion is done in RealRobotAdapter
    
    def _convert_obs(self, obs):

        return obs  # Conversion is done in RealRobotAdapter

    def _send_step_request(self, action: np.ndarray):

        flags = 0

        # Determine action metadata
        action_metadata = {
            "dtype": str(action.dtype),
            "shape": action.shape,
        }
        
        # Send request and metadata
        request = {
            "command": "step",
            "action_metadata": action_metadata
        }
        self._socket.send_json(request, flags | zmq.SNDMORE)

        # Send action data
        self._socket.send(action, flags, copy=False, track=False)

    def _send_reset_request(self):

        flags = 0

        # Send request
        request = {"command": "reset"}
        self._socket.send_json(request, flags)

    def _send_get_observation_request(self):

        flags = 0

        # Send request
        request = {"command": "get_observation"}
        self._socket.send_json(request, flags)

    def _receive_results(self) -> Dict[str, Union[str, int, float, bool, np.ndarray, Dict[str, str]]]:

        flags = 0

        # Receive metadata and simple results
        results = self._socket.recv_json(flags)

        obs_metadata = results.pop("observation_metadata")

        # Receive observation
        data_parts = self._socket.recv_multipart(flags, copy=False, track=False)

        # Reconstruct observation
        obs = {}
        for (key, metadata), data in zip(obs_metadata.items(), data_parts):
            buf = memoryview(data)
            array = np.frombuffer(buf, dtype=metadata["dtype"]).reshape(metadata["shape"])
            obs[key] = array
        results["observation"] = obs

        return results
