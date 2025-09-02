import einops
import numpy as np
import requests
import json_numpy

json_numpy.patch()

# action = requests.post(
#     "http://0.0.0.0:8000/act",
#     json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
# ).json()

# for kitchen it takes 224,224

def from_standard_obs(std_obs, instr):
    cf_obs = {
        "image": std_obs["primary_camera"],
        "instruction": instr,
    }

    return cf_obs

class OpenVLAClient:
    def __init__(self, server_url="http://0.0.0.0:8000/act"):
        self.server_url = server_url
        self.instruction = None

    def __call__(self, observation):
        """
        Calls the OpenVLA server's `/act` endpoint to predict an action for a given image and instruction.

        :param image: A numpy array representing the image (e.g., shape (256, 256, 3)).
        :param instruction: A string with the instruction for the model.
        :param server_url: The URL of the OpenVLA API server.
        :return: The predicted action as a numpy array or None if an error occurs.
        """
        assert self.instruction is not None
        
        try:
            payload = from_standard_obs(observation, instr=self.instruction)
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
            
            ret = json_numpy.loads(response.content)

            # for whatever reason openvla returns gripper in [0,1], so just work with I can't bother debugging it
            ret = ret.copy()            
            ret[..., -1] = 1.0 if ret[..., -1] > 0.5 else -1.0

            return ret
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None
        
    def reset(self, text):
        self.instruction = text
        pass