from typing import List, Dict, Any
from datetime import datetime
import json
import asyncio
from pydantic import BaseModel
import re
class BenchmarkResult(BaseModel):
    timestamp: str
    gpu_info: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    score: float

class ReliabilityAgent:
    def __init__(self, hyperbolic_toolkit):
        self.toolkit = hyperbolic_toolkit
        self.tools = {tool.name: tool for tool in self.toolkit.get_tools()}
        self.results: List[BenchmarkResult] = []
        self.current_gpu = None
    def _parse_gpu_response(self, raw_response: str) -> dict:
            gpus = []
            
            # split response 
            gpu_entries = raw_response.split("----------------------------------------")
            
            for entry in gpu_entries:
                if not entry.strip():
                    continue
                    
                # extract key gpu info 
                cluster_match = re.search(r"Cluster: (.+)", entry)
                node_match = re.search(r"Node ID: (.+)", entry)
                model_match = re.search(r"GPU Model: (.+)", entry)
                available_match = re.search(r"Available GPUs: (\d+)/(\d+)", entry)
                price_match = re.search(r"Price: \$(\d+\.\d+)/hour", entry)
                
                if all([cluster_match, node_match, model_match, available_match, price_match]):
                    gpus.append({
                        "cluster_name": cluster_match.group(1).strip(),
                        "node_name": node_match.group(1).strip(),
                        "gpu_model": model_match.group(1).strip(),
                        "available_count": int(available_match.group(1)),
                        "total_count": int(available_match.group(2)),
                        "price_per_hour": float(price_match.group(1))
                    })
            
            return {"gpus": gpus}

    async def _wait_for_instance(self, timeout_seconds: int = 120):
        # waits for instance to be online 
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout_seconds:
            status_response = await self.tools["get_gpu_status"].arun("")
            instance = self._get_current_instance(status_response)
            
            print(f"Instance status: {instance['instance']['status']}")
            
            if instance['instance']['status'] == 'online':
                self.current_gpu = instance
                return True
                
            await asyncio.sleep(10)
            
        return False

    async def benchmark_cycle(self):
            # run one complete benchmark per cycle
            try:
                print("Getting available GPUs...")
                raw_response = await self.tools["get_available_gpus"].arun("")
                
                gpus_data = self._parse_gpu_response(raw_response)
                if not gpus_data["gpus"]:
                    print("No GPUs found in response")
                    return
                    
                print("\nSelecting GPU...")
                selected_gpu = self._select_gpu(gpus_data["gpus"])
                print(f"Selected GPU: {json.dumps(selected_gpu, indent=2)}")
                
                print("\nRenting GPU...")
                rent_response = await self.tools["rent_compute"].arun({
                    "cluster_name": selected_gpu["cluster_name"],
                    "node_name": selected_gpu["node_name"],
                    "gpu_count": "1"
                })
                print(f"Rent response: {rent_response}")
                
                print("\nWaiting for instance to start...")
                await asyncio.sleep(15)
                
                print("\nGetting instance status...")
                status_response = await self.tools["get_gpu_status"].arun("")
                # print(f"Raw status response: {status_response}")  
                
                self.current_gpu = self._get_current_instance(status_response)
                print(f"Current instance: {json.dumps(self.current_gpu, indent=2)}")
                
                if not self.current_gpu.get('id'):
                    raise ValueError("Failed to get instance ID from status")
                print("\nWaiting for instance to be ready...")
                if not await self._wait_for_instance(timeout_seconds=180):
                    raise TimeoutError("Instance failed to reach running state")
                
                print("\nInstance is ready, running benchmarks...")
                benchmark_results = await self._run_benchmarks()
                
                # calculate & store the benchmarking results
                score = self._calculate_score(benchmark_results)
                self.results.append(BenchmarkResult(
                    timestamp=datetime.now().isoformat(),
                    gpu_info=self.current_gpu,
                    benchmark_results=benchmark_results,
                    score=score
                ))
                
                    
            except Exception as e:
                print(f"Error during benchmark cycle: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                await self._cleanup()

    async def _run_benchmarks(self) -> Dict[str, Any]:
        # run the benchmakring suite after sshing into the machine
        try:
            # Get SSH connection details
            ssh_host = self.current_gpu["sshCommand"].split("@")[1].split()[0]
            ssh_port = int(self.current_gpu["sshCommand"].split("-p")[1].strip())
            
            print(f"\nConnecting to SSH - Host: {ssh_host}, Port: {ssh_port}")
            
            # Connect SSH
            ssh_response = await self.tools["ssh_connect"].arun({
                "host": ssh_host,
                "username": "ubuntu",
                "port": ssh_port
            })
            print(f"SSH Connection Response: {ssh_response}")
            
            results = {}

            # currently naively executes becnhamrking commands 
            commands = [
                "nvidia-smi",
                "python3 -c 'import torch; print(torch.__version__)'",
                "git clone https://github.com/pytorch/examples.git",
                "cd examples/mnist && python main.py --epochs 1"
            ]
            
            for cmd in commands:
                print(f"\nExecuting: {cmd}")
                results[cmd] = await self.tools["remote_shell"].arun({
                    "command": cmd
                })
                print(f"Result: {results[cmd]}")
                
            return results
            
        except Exception as e:
            print(f"Error running benchmarks: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
    def _select_gpu(self, gpus: List[Dict]) -> Dict:
        # selects the best gpu based on defined criteria (rn it's available_count)
        available = [gpu for gpu in gpus if gpu["available_count"] > 0]
        return min(available, key=lambda x: x["price_per_hour"])
        
    def _get_current_instance(self, status_response) -> Dict:
        # get the current instance after using the status tool 
        if isinstance(status_response, str):
            status = json.loads(status_response)
        else:
            status = status_response
            
        # print(f"Full status response: {json.dumps(status, indent=2)}")
        
        if not isinstance(status, dict) or 'instances' not in status:
            raise ValueError(f"Unexpected status format: {status}")
            
        instances = status['instances']
        if not instances:
            raise ValueError("No instances found in status response")
            
        # Get most recent instance
        instance = sorted(
            instances,
            key=lambda x: x.get('start', ''),
            reverse=True
        )[0]
        
        # Make sure we have an instance ID
        if 'id' not in instance:
            raise ValueError(f"Instance missing ID: {instance}")
            
        return instance
    
    def _calculate_score(self, results: Dict) -> float:
        # calcualtes a reliability score 
        # TODO: implement 
        return 0.0
        
    async def _cleanup(self):
        # cleanup the resource (terminate tool)
        if self.current_gpu:
            try:
                print("\nCleaning up - terminating instance...")
                
                # Get instance ID from the correct path in the data structure
                instance_id = self.current_gpu["instance"]["id"]
                print(f"Terminating instance: {instance_id}")
                
                await self.tools["terminate_compute"].arun({
                    "instance_id": instance_id
                })
                print("Instance terminated successfully")
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                import traceback
                traceback.print_exc()