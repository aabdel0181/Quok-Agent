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
- The action will read the readme to understand how to install base dependencies and run the gpu_health.sh script
- The health check script will analyze the GPU’s temperature, memory usage, power draw, and other key metrics.
- The action will classify the GPU as either 'flagged' (if there are concerns) or 'healthy' (if there are no major issues).
- The health data will be stored in the GPUHealthChecks DynamoDB table using the dynamodb_inserter tool. 
"""

class GPUHealthCheckInput(BaseModel):
    """Input argument schema for GPU health check action."""
    data_store: Optional[str] = Field(
        "gpu_health_data_NEW.json",
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
    # gpu_check_script = "gpu_health.sh"
    # repo_name = os.getenv("REPO_NAME")
    # repo_url = os.getenv("REPO_URL")
    repo_name = "Quok-benchmark"
    repo_url = "https://github.com/aabdel0181/Quok-benchmark.git"

    # if not repo_name or not repo_url:
    #     return json.dumps({"error": "Missing REPO_NAME or REPO_URL in .env file"}, indent=2)
    def check_dependencies():
        """Ensure required dependencies are available."""
        print("Checking dependencies...")
        # Check for bc and jq (required by the script)
        for pkg in ['bc', 'jq']:
            result = execute_remote_command(f"which {pkg}")
            if not result.strip():
                print(f"Installing {pkg}...")
                execute_remote_command(f"sudo apt-get update && sudo apt-get install -y {pkg}")

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
        output = execute_remote_command(f"cd {repo_name} && sudo chmod +x gpu_health.sh")
        print(f"Cloned repository contents: {output}")

    def install_dependencies():
            """Install required dependencies for the benchmark suite."""
            print("Installing dependencies...")
            execute_remote_command(f"cd {repo_name} && python3 main.py --install")

    def run_health_check() -> dict:
        """Run the GPU health check script and parse results."""
        print("Running GPU health check...")
        try:
            # Execute the health check script
            
             # First, change to the scripts directory and update permissions on the script
            chmod_command = f"cd {repo_name}/scripts && chmod +x gpu_health.sh"
            chmod_output = execute_remote_command(chmod_command)
            print("chmod output:", chmod_output)
            
            # Then, run the script using sudo
            run_command = f"cd {repo_name}/scripts && sudo ./gpu_health.sh"
            result = execute_remote_command(run_command)
            print("result: ", result)
            health_data = json.loads(result)
            
            # Add timestamp and determine if flagged
            health_data.update({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "flagged": health_data.get("overall_status") != "healthy"
            })
            
            return health_data
            
        except Exception as e:
            print(f"Error running health check: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "gpu_uuid": get_gpu_uuid()
            }
        
    def get_gpu_uuid() -> str:
        """Retrieve GPU UUID from the system."""
        uuid_output = execute_remote_command("nvidia-smi --query-gpu=uuid --format=csv,noheader")
        return uuid_output.strip().split('\n')[0] if uuid_output else "unknown_gpu"

    def store_data(health_data: dict):
        """Store GPU health data in the specified location."""
        gpu_uuid = health_data.get("gpu_info", {}).get("uuid", "unknown_gpu")
        log_folder = os.path.join("gpu_logs", gpu_uuid)
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, data_store)

        try:
            existing_data = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    existing_data = json.load(f)
            existing_data.append(health_data)
            with open(log_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            print(f"GPU health data stored in {log_file}")
        except Exception as e:
            print(f"Error storing GPU health data: {e}")

    # Main execution flow
    try:
        if not repo_exists():
            clone_repo()
        
        check_dependencies()
        health_data = run_health_check()
        store_data(health_data)
        
        return json.dumps({
            "gpu_health": health_data,
            "status": "flagged" if health_data.get("flagged", False) else "healthy"
        }, indent=2)
        
    except Exception as e:
        error_data = {
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "gpu_uuid": get_gpu_uuid()
        }
        store_data(error_data)
        return json.dumps({"error": str(e)}, indent=2)

class GPUHealthCheckAction(HyperbolicAction):
    """GPU health check action."""
    name: str = "gpu_health_check"
    description: str = GPU_HEALTH_CHECK_PROMPT
    args_schema: Optional[type[BaseModel]] = GPUHealthCheckInput
    func: Callable[..., str] = check_gpu_health
