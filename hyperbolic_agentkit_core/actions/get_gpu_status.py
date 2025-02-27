import requests
import json
from typing import Optional
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.utils import get_api_key

GET_GPU_STATUS_PROMPT = """
This tool will get all the the status and ssh commands of you currently rented GPUs on the Hyperbolic platform.

It does not take any inputs

Important notes:
- Authorization key is required for this operation
- The GPU prices are in CENTS per hour
- If the status is "starting", it means the GPU is not ready yet. You can use the GetGPUStatus Action to check the status again after 5 seconds.
- You can access it through the SSHAccess Action and run commands on it through the RemoteShell Action.
"""


class GetGpuStatusInput(BaseModel):
  """Input argument schema for getting available GPUs."""

def get_gpu_status() -> str:
    """
    Returns a string representation of the response from the Hyperbolic API.
    
    Tries up to 5 times to get a successful response from the API.
    Returns:
      A JSON object representing the API response or an error message if all attempts fail.
    """
    api_key = get_api_key()
    url = "https://api.hyperbolic.xyz/v1/marketplace/instances"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    attempts = 0
    last_error = ""
    
    while attempts < 5:
        attempts += 1
        try:
            response = requests.get(url, headers=headers)
            # If successful, return the parsed JSON response.
            if response.status_code == 200:
                return response.json()
            else:
                last_error = response.text
                print(f"Attempt {attempts}: Received status code {response.status_code} with error: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempts} failed with exception: {last_error}")
        
        time.sleep(5)  # Wait 5 seconds between attempts
    
    return {"detail": f"Internal server error after 5 attempts: {last_error}"}


class GetGpuStatusAction(HyperbolicAction):
  """Get status for my GPUs action."""

  name: str = "get_gpu_status"
  description: str = GET_GPU_STATUS_PROMPT
  args_schema: type[BaseModel] | None = GetGpuStatusInput
  func: Callable[..., str] = get_gpu_status
