import os
import sys
import json

# adding the parent directory for hyperbolic toolkit 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import asyncio
from dotenv import load_dotenv
from hyperbolic_langchain.agent_toolkits import HyperbolicToolkit
from hyperbolic_langchain.utils import HyperbolicAgentkitWrapper
from reliability_agent.agent import ReliabilityAgent

async def main():
    # print(f"Python path: {sys.path}")  
    # print(f"Looking for modules in: {parent_dir}")  
    
    # Initialize components
    hyperbolic = HyperbolicAgentkitWrapper()
    toolkit = HyperbolicToolkit.from_hyperbolic_agentkit_wrapper(hyperbolic)
    
    # Create reliability agent with the existing hyperbolic toolkit
    agent = ReliabilityAgent(toolkit)
    
    # Run benchmark cycle
    await agent.benchmark_cycle()
    
    # Print results
    for result in agent.results:
        print(json.dumps(result.dict(), indent=2))

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())