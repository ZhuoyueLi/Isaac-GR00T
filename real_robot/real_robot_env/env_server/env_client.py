from typing import Dict, Tuple, Union
import gymnasium as gym
import numpy as np
import zmq

class EnvClient(gym.Env):

    """
    
    This class is a gym environment that communicates with an EnvServer to implement its methods.
    The purpose of this class is to serve as a template that can be copied and adapted to a different project.

    Example Usage:

        from real_robot_env.env_server.env_client import EnvClient
        import numpy as np

        client = EnvClient(
            host = "127.0.0.1",
            port = 6060,
        )

        assert client.connect()

        obs, info = client.reset()
        obs, reward, term, trunc, info = client.step(np.array([0.2719, -0.5165, 0.2650, -1.6160, -0.0920, 1.6146, -1.7760, -1]))

        client.close()
    
    """

    def __init__(self, name = "Environment Client", host: str = "127.0.0.1", port: int = 6060):

        self.name = name
        self.host = host
        self.port = port
        self._addr = f"tcp://{self.host}:{self.port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

        # TODO Add action + observation space

    def connect(self) -> bool:

        print(f"Connecting to {self.name}...")
        try:
            self._socket.connect(self._addr)
            print("Success")
            return True
        except Exception as e:
            print("Failed with exception: ", e)
            return False

    def close(self) -> bool:

        self._socket.disconnect(self._addr)
        print(f"Closed connection to {self.name}")
        return True

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, str]]:
            
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_step_request(action)
        
        # Receive response
        results = self._receive_results()

        return results["observation"], results["reward"], results["terminated"], results["truncated"], results["info"]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_reset_request()
        
        # Receive response
        results = self._receive_results()

        return results["observation"], results["info"]
    
    def get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        
        # TODO If you want to use this method, make sure the environment that you are using supports it
        
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_get_observation_request()
        
        # Receive response
        results = self._receive_results()

        return results["observation"]

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
