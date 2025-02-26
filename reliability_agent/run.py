import os
import sys
import json
from contextlib import asynccontextmanager

# adding the parent directory for hyperbolic toolkit 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import asyncio
from dotenv import load_dotenv
from hyperbolic_langchain.agent_toolkits import HyperbolicToolkit
from hyperbolic_langchain.utils import HyperbolicAgentkitWrapper
from reliability_agent.agent import ReliabilityAgent

@asynccontextmanager
async def manage_agent(toolkit):
    """Context manager to handle agent lifecycle and ensure conversation is saved."""
    agent = ReliabilityAgent(toolkit)
    try:
        yield agent
    finally:
        agent.save_conversation()

async def main():
    # Initialize components
    hyperbolic = HyperbolicAgentkitWrapper()
    toolkit = HyperbolicToolkit.from_hyperbolic_agentkit_wrapper(hyperbolic)
    
    # Use context manager to ensure conversation is saved
    async with manage_agent(toolkit) as agent:
        # Run benchmark cycle
        await agent.benchmark_cycle()
        
        # Print results
        for result in agent.results:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())