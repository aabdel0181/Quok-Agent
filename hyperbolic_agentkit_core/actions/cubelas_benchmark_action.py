import os
import json
import datetime
from typing import Optional, Callable
from collections.abc import Callable

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from hyperbolic_agentkit_core.actions.dynamodb_inserter import DynamoDBInsertAction

from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.remote_shell import RemoteShellAction, RemoteShellInput
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command

# Load environment variables from .env
load_dotenv()

CUBLAS_BENCHMARK_PROMPT = """
This tool runs CUBLAS benchmarks on the machine it is running on. The process follows these steps:

1. Verify that the benchmark repository exists; if not, clone it
2. Install CUBLAS dependencies using the install script
3. Make the benchmark script executable
4. Run the CUBLAS benchmark
5. Parse and return the results from the output file
6. Store the benchmark results with timestamp and metadata
7. Store the data in DynamoDB with different table: 
"# First run CUBLAS benchmark
benchmark_result = await agent.run_tool("cublas_benchmark")

# Then store in DynamoDB with different table
await agent.run_tool("dynamodb_insert", {
    "table_name": "CUBLASBenchmarks",
    "data": benchmark_result
})"

The tool will return the benchmark results in a structured JSON format.
"""

class CUBLASBenchmarkInput(BaseModel):
    """Input argument schema for CUBLAS benchmark action."""
    data_store: Optional[str] = Field(
        "cublas_benchmark_data.json",
        description="The file where benchmark data will be stored"
    )

def run_cublas_benchmark(data_store: str = "cublas_benchmark_data.json") -> str:
    """
    Runs CUBLAS benchmark after verifying repository and dependencies.
    
    Args:
        data_store (str): The file where benchmark data will be stored.
    
    Returns:
        str: A formatted JSON string with benchmark results.
    """
    repo_name = "Quok-benchmark"
    repo_url = "https://github.com/aabdel0181/Quok-benchmark.git"

    def repo_exists() -> bool:
        """Check if the repository exists in the current directory."""
        result = execute_remote_command(f"ls | grep {repo_name}")
        if result.strip():
            output = execute_remote_command(f"cd {repo_name} && ls")
            return True
        return False

    def clone_repo():
        """Clone the repository and set up permissions."""
        print(f"Repository '{repo_name}' not found. Cloning from {repo_url}...")
        execute_remote_command(f"git clone {repo_url}")
        # Make scripts executable
        execute_remote_command(f"cd {repo_name} && chmod +x scripts/install/install.sh")
        execute_remote_command(f"cd {repo_name} && chmod +x benchmarks/cublas/run.sh")


    def store_in_dynamodb(benchmark_data: dict) -> str:
        """Store benchmark data in DynamoDB using the existing inserter tool."""
        try:
            # Create DynamoDB inserter with CUBLAS table
            dynamo_inserter = DynamoDBInsertAction()
            
            
            # Use the existing tool to insert data
            response = dynamo_inserter.func(
                table_name="CUBLASBenchmarks",
                data=benchmark_data
            )
            
            return response
            
        except Exception as e:
            return f"Error inserting data into DynamoDB: {str(e)}"
    def install_dependencies():
        """Install CUBLAS dependencies."""
        print("Installing CUBLAS dependencies...")
        execute_remote_command(f"cd {repo_name} && sudo ./scripts/install/install.sh --cublas")

    def run_benchmark() -> dict:
        """Run the CUBLAS benchmark and parse results."""
        print("Running CUBLAS benchmark...")
        try:
            # Run the benchmark
            execute_remote_command(f"cd {repo_name}/benchmarks/cublas && sudo ./run.sh")
            
            # Read the results file
            result = execute_remote_command(f"cat {repo_name}/results/cublas_raw.txt")
            
            # Parse the raw output into structured data
            benchmark_data = {
                "Timestamp": int(datetime.datetime.utcnow().timestamp()),
                "GPU_UUID": get_gpu_uuid(),
                "raw_output": result,
                "parsed_results": parse_benchmark_output(result)
            }
            
            return benchmark_data
            
        except Exception as e:
            print(f"Error running CUBLAS benchmark: {e}")
            return {
                "error": str(e),
                "Timestamp": int(datetime.datetime.utcnow().timestamp()),
                "GPU_UUID": get_gpu_uuid()
            }


    def get_gpu_uuid() -> str:
        """Retrieve GPU UUID from the system."""
        uuid_output = execute_remote_command("nvidia-smi --query-gpu=uuid --format=csv,noheader")
        return uuid_output.strip().split('\n')[0] if uuid_output else "unknown_gpu"

    def parse_benchmark_output(raw_output: str) -> dict:
        """Parse the benchmark output into structured data."""
        # Initialize results structure
        results = {
            "performance_metrics": [],
            "summary": {}
        }
        
        try:
            # Split output into lines and parse
            lines = raw_output.strip().split('\n')
            for line in lines:
                # Add parsing logic based on the actual output format
                # This is a placeholder - adjust based on actual output format
                if line.startswith("TEST"):
                    results["performance_metrics"].append(line)
                elif line.startswith("Average"):
                    results["summary"]["average"] = line
                    
            return results
        except Exception as e:
            print(f"Error parsing benchmark output: {e}")
            return {"error": "Failed to parse benchmark output"}

    def store_data(benchmark_data: dict):
        """Store benchmark data in the specified location."""
        gpu_uuid = benchmark_data.get("gpu_uuid", "unknown_gpu")
        log_folder = os.path.join("benchmark_logs", gpu_uuid)
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, data_store)

        try:
            existing_data = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    existing_data = json.load(f)
            existing_data.append(benchmark_data)
            with open(log_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            print(f"Benchmark data stored in {log_file}")
        except Exception as e:
            print(f"Error storing benchmark data: {e}")

    # Main execution flow
    try:
        if not repo_exists():
            clone_repo()
        
        install_dependencies()
        benchmark_data = run_benchmark()
        store_data(benchmark_data)
        
        return json.dumps(benchmark_data, indent=2)
        
    except Exception as e:
        error_data = {
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "gpu_uuid": get_gpu_uuid()
        }
        store_data(error_data)
        dynamo_response = store_in_dynamodb(benchmark_data)
        print(dynamo_response)

        return json.dumps({"error": str(e)}, indent=2)

class CUBLASBenchmarkAction(HyperbolicAction):
    """CUBLAS benchmark action."""
    name: str = "cublas_benchmark"
    description: str = CUBLAS_BENCHMARK_PROMPT
    args_schema: Optional[type[BaseModel]] = CUBLASBenchmarkInput
    func: Callable[..., str] = run_cublas_benchmark