import numpy as np
import requests
import json_numpy
import einops

# crossformer and octo use the same obs format 
from .crossformer_client import from_standard_obs

json_numpy.patch()

# clients expect standardized obs, and output standardized actions, see rr_client

class OctoClient:
    def __init__(self, server_url: str = "http://0.0.0.0:8001", ensemble=False):
        self.server_url = server_url
        self.ensemble = ensemble

    def __call__(self, observation: dict):
        """
        Calls the Octo server's `/query` endpoint to predict an action for a given observation.

        :param observation: A dictionary representing the observation data.
        :return: The predicted action as a numpy array or None if an error occurs.
        """
        try:
            observation = from_standard_obs(observation)
            payload = {"observation": observation, "ensemble": self.ensemble}
            response = requests.post(f"{self.server_url}/query", json=payload)
            response.raise_for_status()
            
            retval = json_numpy.loads(response.json())

            if len(retval.shape) == 2: # horizon, dim
                return retval[0]
            else:
                return retval
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None

    def reset(self, text: str):
        """
        Calls the Octo server's `/reset` endpoint to reset the task with a new text instruction.

        :param text: A string representing the task description.
        :return: Response status message or None if an error occurs.
        """
        try:
            payload = {"text": text}
            response = requests.post(f"{self.server_url}/reset", json=payload)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None


# Example usage:
if __name__ == "__main__":
    client = OctoClient()
    
    dummy_observation = {
        "proprio_bimanual": [0.0] * 14,
        "image_high": [[[0] * 3] * 224] * 224,
        "image_left_wrist": [[[0] * 3] * 224] * 224,
        "image_right_wrist": [[[0] * 3] * 224] * 224,
    }
    action = client(dummy_observation)
    print("Predicted action:", action)
    
    reset_response = client.reset("New task instruction")
    print("Reset response:", reset_response)
