import os
import sys
import json
import random
from datetime import datetime

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from hyperbolic_langchain.agent_toolkits import HyperbolicToolkit
from hyperbolic_langchain.utils import HyperbolicAgentkitWrapper
import asyncio
from dotenv import load_dotenv

class ReliabilityTester:
    def __init__(self, toolkit):
        self.toolkit = toolkit
        self.tools = {tool.name: tool for tool in toolkit.get_tools()}
        self.current_gpu = None
        
    async def run_single_test(self):
        try:
            print("\n1. Getting available GPUs...")
            # Debug: Print the actual tool
            gpu_tool = self.tools.get("get_available_gpus")
            print(f"GPU Tool: {gpu_tool}")
            
            # Debug: Print raw response
            raw_response = await self.tools["get_available_gpus"].arun("")
            print(f"Raw Response: {raw_response}")
            
            # Only try to parse if we got a response
            if raw_response:
                gpus = json.loads(raw_response)
                print(f"Found GPUs: {json.dumps(gpus, indent=2)}")
            else:
                print("No response from get_available_gpus tool")
                return
                
        except KeyError as e:
            print(f"Tool not found error: {e}")
            print(f"Available tools: {list(self.tools.keys())}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response was: {raw_response}")
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
        
        # Print available tools
        print("\nAvailable tools:")
        tools = hyperbolic_toolkit.get_tools()
        for tool in tools:
            print(f"- {tool.name}: {type(tool)}")
            
        # Create and run tester
        tester = ReliabilityTester(hyperbolic_toolkit)
        print("\nStarting benchmark test...")
        await tester.run_single_test()
            
    except Exception as e:
        print(f"Error running reliability agent: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Debug: Print environment variables (without the actual API key value)
    env_vars = {
        "HYPERBOLIC_API_KEY": "✓" if os.getenv("HYPERBOLIC_API_KEY") else "✗",
        "SSH_PRIVATE_KEY_PATH": os.getenv("SSH_PRIVATE_KEY_PATH"),
        "PYTHONPATH": os.getenv("PYTHONPATH")
    }
    print("Environment variables:")
    for key, value in env_vars.items():
        print(f"{key}: {value}")
    
    # Run the agent
    asyncio.run(main())