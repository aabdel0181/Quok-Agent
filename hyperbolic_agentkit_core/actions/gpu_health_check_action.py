import os
import json
import datetime
from typing import Optional, Callable
from collections.abc import Callable

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.remote_shell import RemoteShellAction, RemoteShellInput
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command

# Load environment variables from .env
load_dotenv()

GPU_HEALTH_CHECK_PROMPT = """
This tool will check the GPU health of the machine it is running on.

It takes the following inputs:
- data_store: The file where GPU health data will be stored.

Important notes:
- Repository information is automatically retrieved from the .env file.
- If the repository does not exist, the action will clone it before running the health check.
- The health check script will analyze the GPU’s temperature, memory usage, power draw, and other key metrics.
- The action will classify the GPU as either 'flagged' (if there are concerns) or 'healthy' (if there are no major issues).
- The health data will be stored in a JSON file for tracking and future analysis.
"""

class GPUHealthCheckInput(BaseModel):
    """Input argument schema for GPU health check action."""
    data_store: Optional[str] = Field(
        "gpu_health_data.json",
        description="The file where GPU health data will be stored"
    )

def check_gpu_health(data_store: str = "gpu_health_data.json") -> str:
    """
    Checks GPU health after verifying if a repository exists.
    
    Args:
        data_store (str): The file where GPU health data will be stored.
    
    Returns:
        str: A formatted JSON string with GPU health classification.
    """
    gpu_check_script = "gpu_health.sh"
    repo_name = os.getenv("REPO_NAME")
    repo_url = os.getenv("REPO_URL")

    if not repo_name or not repo_url:
        return json.dumps({"error": "Missing REPO_NAME or REPO_URL in .env file"}, indent=2)

    def get_gpu_uuid() -> str:
        """Retrieve GPU UUID from the system."""
        uuid_output = execute_remote_command("nvidia-smi --query-gpu=uuid --format=csv,noheader")
        return uuid_output.strip().split('\n')[0] if uuid_output else "unknown_gpu"

    def repo_exists() -> bool:
        """Check if the repository exists in the current directory via SSH."""
        result = execute_remote_command(f"ls | grep {repo_name}")
        if result.strip():
            output = execute_remote_command(f"cd {repo_name} && ls")
            print(f"Inside the repo contents: {output}")
            return True
        return False

    def clone_repo():
        """Clone the repository if it does not exist via SSH."""
        print(f"Repository '{repo_name}' not found. Cloning from {repo_url}...")
        execute_remote_command(f"git clone {repo_url}")
        output = execute_remote_command(f"cd {repo_name} && ls")
        print(f"Cloned repository contents: {output}")

    def run_gpu_health_check():
        """Run the GPU health check script and return results as a dictionary."""
        permission_check = f"cd {repo_name} && chmod +x {gpu_check_script}"
        execute_remote_command(permission_check)

        command = f"cd {repo_name} && bash {gpu_check_script}"
        output = execute_remote_command(command)

        print("Raw GPU Health Check Output:")
        print(output)

        if not output:
            print("No output from GPU health check script.")
            return {"error": "GPU health check script produced no output"}
        
        return parse_health_check_output(output)

    def parse_health_check_output(output: str):
        """Parses GPU health check output into a structured dictionary."""
        gpu_data = {"concerns": [], "fine": [], "flagged": False}
        capturing_section = None

        for line in output.split("\n"):
            line = line.strip()
            if "Concerns detected" in line:
                capturing_section = "concerns"
            elif "Fine status" in line:
                capturing_section = "fine"
            elif "--------------------------" in line:
                capturing_section = None
            elif capturing_section:
                gpu_data[capturing_section].append(line)
        
        gpu_data["flagged"] = bool(gpu_data["concerns"])
        gpu_data["uuid"] = get_gpu_uuid()
        gpu_data["timestamp"] = datetime.datetime.utcnow().isoformat()
        return gpu_data

    def store_data(gpu_data):
        """Store GPU health data in a JSON file for tracking."""
        gpu_uuid = gpu_data.get("uuid", "unknown_gpu")
        log_folder = os.path.join("gpu_logs", gpu_uuid)
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, data_store)

        try:
            existing_data = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    existing_data = json.load(f)
            existing_data.append(gpu_data)
            with open(log_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            print(f"GPU health data stored successfully in {log_file}")
        except Exception as e:
            print(f"Error storing GPU health data: {e}")

    # Ensure repo exists, otherwise clone it
    if not repo_exists():
        clone_repo()
    
    print("Running GPU health check...")
    gpu_data = run_gpu_health_check()
    store_data(gpu_data)

    return json.dumps({
        "gpu_health": gpu_data,
        "status": "flagged" if gpu_data["flagged"] else "healthy"
    }, indent=2)

class GPUHealthCheckAction(HyperbolicAction):
    """GPU health check action."""
    name: str = "gpu_health_check"
    description: str = GPU_HEALTH_CHECK_PROMPT
    args_schema: Optional[type[BaseModel]] = GPUHealthCheckInput
    func: Callable[..., str] = check_gpu_health
