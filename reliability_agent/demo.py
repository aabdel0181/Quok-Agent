# Script to get familiar with the hyperbolictoolkit (as a script, not yet an agent)
import os
import sys
import json
import random
from datetime import datetime
import re

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from hyperbolic_langchain.agent_toolkits import HyperbolicToolkit
from hyperbolic_langchain.utils import HyperbolicAgentkitWrapper
import asyncio
from dotenv import load_dotenv

def parse_gpu_response(raw_response: str) -> dict:
    # parse the string into a dictionary
    gpus = []
    
    # split the string into individual GPU entries
    gpu_entries = raw_response.split("----------------------------------------")
    
    for entry in gpu_entries:
        if not entry.strip():
            continue
            
        # get info  using regex
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

class ReliabilityTester:
    def __init__(self, toolkit):
        self.toolkit = toolkit
        self.tools = {tool.name: tool for tool in toolkit.get_tools()}
        self.current_gpu = None
        
    async def run_single_test(self):
        try:
            
            # print("\n1. Getting available GPUs...")
            # raw_response = await self.tools["get_available_gpus"].arun("")
            
            # # Parse the string response into a json
            # gpus_data = parse_gpu_response(raw_response)
            # print(f"\nParsed GPU Data: {json.dumps(gpus_data, indent=2)}")
            
            # if not gpus_data["gpus"]:
            #     print("No GPUs available!")
            #     return
                
            # print("\n2. Selecting a GPU...")
            # # Filter for available GPUs and sort by price
            # available_gpus = [
            #     gpu for gpu in gpus_data["gpus"] 
            #     if gpu["available_count"] > 0
            # ]
            
            # if not available_gpus:
            #     print("No GPUs currently available!")
            #     return
                
            # # Select cheapest GPU for testing
            # selected_gpu = min(available_gpus, key=lambda x: x["price_per_hour"])
            # print(f"Selected GPU: {json.dumps(selected_gpu, indent=2)}")
            
            # print("\n3. Attempting to rent GPU...")
            # rent_response = await self.tools["rent_compute"].arun({
            #     "cluster_name": selected_gpu["cluster_name"],
            #     "node_name": selected_gpu["node_name"],
            #     "gpu_count": "1"
            # })
            # print(f"Rent Response: {rent_response}")
                        
            # print("\n4. Waiting for instance to start...")
            # await asyncio.sleep(15)  # Give instance time to start
            
            print("\n5. Getting instance status...")
            status_response = await self.tools["get_gpu_status"].arun("")
            
            instance_status = status_response
                
            print(f"Instance Status: {json.dumps(instance_status, indent=2)}")
            
            # Check if we have any instances
            if not instance_status.get('instances'):
                print("No instances found in status response")
                return
                
            # Find our most recently starte instance 
            current_instance = sorted(
                instance_status['instances'],
                key=lambda x: x['start'],
                reverse=True
            )[0]
            
            print(f"\nFound instance: {current_instance['id']}")
            self.current_gpu = current_instance
            
            # Check instance status
            if current_instance['instance']['status'] == 'starting':
                print("Instance is still starting, waiting longer...")
                await asyncio.sleep(30)
            print("\n6. Establishing SSH connection...")
            # Print available tools for debugging
            print("Available tools:", list(self.tools.keys()))
            
            ssh_response = await self.tools["ssh_connect"].arun(json.dumps({
                "host": self.current_gpu["sshCommand"].split("@")[1].split()[0],
                "username": "ubuntu"
            }))
            print(f"SSH Response: {ssh_response}")

        except Exception as e:
            print(f"Error during testing: {e}")
            print(f"Error type: {type(e)}")

async def main():
    """Main entry point for running the reliability agent"""
    try:
        print("Initializing Reliability Agent...")
        
        # Initialize Hyperbolic components
        hyperbolic_agentkit = HyperbolicAgentkitWrapper()
        hyperbolic_toolkit = HyperbolicToolkit.from_hyperbolic_agentkit_wrapper(hyperbolic_agentkit)
        
        # Create and run tester
        tester = ReliabilityTester(hyperbolic_toolkit)
        print("\nStarting benchmark test...")
        await tester.run_single_test()
            
    except Exception as e:
        print(f"Error running reliability agent: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the agent
    asyncio.run(main())