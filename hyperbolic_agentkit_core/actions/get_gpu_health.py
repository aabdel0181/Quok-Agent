import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager
import json
from collections.abc import Callable
from datetime import datetime

TELEMETRY_PROMPT = """
Pull a GitHub repository containing telemetry benchmarking code, run the benchmarks, and log the results.

Input parameters:
- repo_url: URL of the GitHub repository containing benchmark code
- branch: Branch to use (default: main)
- benchmark_script: Path to the benchmark script in the repo
- output_path: Where to save the telemetry logs
- gpu_info: Current GPU information for context

The action will:
1. Clone/pull the repository
2. Run the benchmark script
3. Collect and log telemetry data
4. Save results to specified output location
"""

class TelemetryInput(BaseModel):
    """Input argument schema for telemetry benchmarking."""
    repo_url: str = Field(..., description="GitHub repository URL containing benchmark code")
    branch: str = Field("main", description="Git branch to use")
    benchmark_script: str = Field(..., description="Path to benchmark script in repo")
    output_path: str = Field(..., description="Path to save telemetry logs")
    gpu_info: Dict[str, Any] = Field(..., description="Current GPU information")

async def run_telemetry_benchmark(
    repo_url: str,
    branch: str = "main",
    benchmark_script: str = "benchmark.py",
    output_path: str = "telemetry_logs",
    gpu_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run telemetry benchmarks and log results.
    
    Args:
        repo_url: GitHub repository URL
        branch: Git branch to use
        benchmark_script: Path to benchmark script
        output_path: Where to save logs
        gpu_info: GPU information
    
    Returns:
        Dict containing benchmark results and metadata
    """
    try:
        # Clone/pull repository
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        clone_cmd = f"git clone {repo_url} -b {branch} || (cd {repo_name} && git pull)"
        await ssh_manager.run_command(clone_cmd)
        
        # Run benchmark script
        benchmark_cmd = f"cd {repo_name} && python {benchmark_script}"
        benchmark_output = await ssh_manager.run_command(benchmark_cmd)
        
        # Collect GPU metrics during benchmark
        gpu_metrics = await ssh_manager.run_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader")
        
        # Structure results
        timestamp = datetime.now().isoformat()
        results = {
            "timestamp": timestamp,
            "gpu_info": gpu_info,
            "benchmark_output": benchmark_output,
            "gpu_metrics": gpu_metrics,
            "metadata": {
                "repo": repo_url,
                "branch": branch,
                "script": benchmark_script
            }
        }
        
        # Save results
        os.makedirs(output_path, exist_ok=True)
        output_file = f"{output_path}/telemetry_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        return {"error": str(e)}

class TelemetryAction(HyperbolicAction):
    """Telemetry benchmarking action."""
    
    name: str = "run_telemetry"
    description: str = TELEMETRY_PROMPT
    args_schema: type[BaseModel] = TelemetryInput
    func: Callable[..., Dict[str, Any]] = run_telemetry_benchmark