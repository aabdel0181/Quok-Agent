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
This tool performs a comprehensive GPU health check on the machine it is running on as part of a multi-step benchmarking workflow. The process must follow these steps in strict order:

1. Verify that the benchmark repository (as specified in the .env file) exists; if not, clone it.
2. Check and install any required dependencies.
3. Navigate to the repository's "scripts" directory, update file permissions (using 'chmod +x') on the GPU health script (gpu_health.sh), and execute the script using sudo.
4. Parse the scripts output, which is expected to be a JSON object containing key metrics such as temperature, memory usage, GPU utilization, and power draw.
5. Augment the JSON output with a current timestamp and additional metadata (e.g., GPU UUID).
6. Store the resulting health data in a local file as specified by the 'data_store' parameter.
7. **Immediately after local storage, call the 'dynamodb_inserter' tool to insert the GPU health data into the GPUHealthChecks DynamoDB table.**
8. Only after confirming that the data has been successfully inserted into DynamoDB, proceed with running the remainder of the benchmarking tests.

Ensure that every step is executed in sequence with proper error handling and that the dynamodb_inserter tool is explicitly invoked using the output data.
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
            # print(f"Inside the repo contents: {output}")
            return True
        return False

    def clone_repo():
        """Clone the repository if it does not exist via SSH."""
        print(f"Repository '{repo_name}' not found. Cloning from {repo_url}...")
        execute_remote_command(f"git clone {repo_url}")
        output = execute_remote_command(f"cd {repo_name} && sudo chmod +x gpu_health.sh")
        # print(f"Cloned repository contents: {output}")

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
            # print("chmod output:", chmod_output)
            
            # Then, run the script using sudo
            run_command = f"cd {repo_name}/scripts && sudo ./gpu_health.sh"
            result = execute_remote_command(run_command)
            # print("result: ", result)
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


      # Main execution flow
    try:
        if not repo_exists():
            clone_repo()
        
        check_dependencies()
        raw_health_data = run_health_check()
        
        # Transform the data for DynamoDB insertion:
        try:
            # Extract GPU UUID from nested structure
            gpu_uuid = raw_health_data.get("gpu_info", {}).get("uuid")
            if not gpu_uuid:
                raise ValueError("GPU UUID not found in health data")
            
            # Extract timestamp string and convert to Unix timestamp
            timestamp_str = raw_health_data.get("timestamp")
            if not timestamp_str:
                raise ValueError("Timestamp not found in health data")
            dt = datetime.datetime.fromisoformat(timestamp_str)
            unix_timestamp = int(dt.timestamp())
            
            # Remove the 'uuid' from gpu_info and the 'timestamp' from the nested structure
            if "uuid" in raw_health_data.get("gpu_info", {}):
                del raw_health_data["gpu_info"]["uuid"]
            if "timestamp" in raw_health_data:
                del raw_health_data["timestamp"]
            
            final_output = {
                "GPU_UUID": gpu_uuid,
                "Timestamp": unix_timestamp,
                "gpu_health": raw_health_data,
                "status": "flagged" if raw_health_data.get("flagged", False) else "healthy"
            }
        except Exception as e:
            print(f"Error transforming health data for DynamoDB: {e}")
            return json.dumps({"error": f"Error transforming data: {str(e)}"}, indent=2)
        
        return json.dumps(final_output, indent=2)
        
    except Exception as e:
        error_data = {
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "gpu_uuid": get_gpu_uuid()
        }
        return json.dumps({"error": str(e)}, indent=2)

class GPUHealthCheckAction(HyperbolicAction):
    """GPU health check action."""
    name: str = "gpu_health_check"
    description: str = GPU_HEALTH_CHECK_PROMPT
    args_schema: Optional[type[BaseModel]] = GPUHealthCheckInput
    func: Callable[..., str] = check_gpu_health
