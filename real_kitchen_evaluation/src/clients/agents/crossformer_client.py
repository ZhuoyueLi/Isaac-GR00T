import numpy as np
import requests
import json_numpy
import einops

json_numpy.patch()

# clients expect standardized obs, and output standardized actions, see rr_client

def from_standard_obs(std_obs):
    bs = 1
    window_size = 1
    cf_obs = {
        "image_primary": einops.repeat(std_obs["primary_camera"], 'h w c -> h w c'),
        "image_secondary": einops.repeat(std_obs["secondary_camera"], 'h w c -> h w c'),
    }

    return cf_obs

class CrossformerClient:
    def __init__(self, server_url: str = "http://0.0.0.0:8007"):
        self.server_url = server_url

    def __call__(self, observation: dict):
        """
        Calls the CrossFormer server's `/query` endpoint to predict an action for a given observation.

        :param observation: A dictionary representing the observation data.
        :return: The predicted action as a numpy array or None if an error occurs.
        """
        try:
            observation = from_standard_obs(observation)
            payload = {"observation": observation,
                       "modality": "l",
                       "ensemble": True,
                       "model": "crossformer"}
            response = requests.post(f"{self.server_url}/query", json=payload)
            response.raise_for_status()
            
            return json_numpy.loads(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None

    def reset(self, text: str):
        """
        Calls the CrossFormer server's `/reset` endpoint to reset the task with a new text instruction.

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
    client = CrossformerClient()
    
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
