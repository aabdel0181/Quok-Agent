from typing import List, Dict, Any
from datetime import datetime
import json
import asyncio
from pydantic import BaseModel
import re, sys
import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage  
from reliability_agent.logging_manager import ConversationLogger, BenchmarkStatus # add our logging manager

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

from utils import print_system, print_ai, print_error, format_ai_message_content, run_with_progress

class ReliabilityAgent:
    def __init__(self, hyperbolic_toolkit):
        self.toolkit = hyperbolic_toolkit
        self.tools = {tool.name: tool for tool in self.toolkit.get_tools()}
        self.results = []
        self.current_gpu = None
        # self.gpu_selector = GPUSelector()
    
        # initialize claude 
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0
        )
        self.logger = ConversationLogger(llm=self.llm) # now we pass the LLM instance; I wonder if it'll work ;P 

        # create REACT agent with correct initialization
        system_message = """You are an AI agent specialized in decentralized GPU reliability assessments and benchmarking.
        You have access to tools for managing GPU instances and running commands.
        Use these tools to set up and run benchmarks efficiently, documenting any errors along the way. Please note that you are non-interactive."""
        
        self.agent_executor = create_react_agent(
            self.llm,
            tools=list(self.tools.values())
        )
        
        # Create runnable config
        self.runnable_config = RunnableConfig(
            recursion_limit=200,
            configurable={
                "system_message": system_message,
                "temperature": 0
            }
        )

    async def _setup_environment(self) -> Dict[str, Any]:
        """Intelligently setup the benchmarking environment."""
        try:
            # Get GPU info first
            print("\nGetting GPU info...")
            gpu_info = await self._get_gpu_info()
            print(f"GPU Info: {json.dumps(gpu_info, indent=2)}")
            
            # Analyze requirements using LLM
            print("\nAnalyzing environment requirements...")
            prompt = f"""Analyze this GPU configuration and determine required setup:

            GPU Information:
            {json.dumps(gpu_info, indent=2)}

            1. First, check if we can access the GPU
            2. Then determine what dependencies we need
            3. Install and validate each dependency
            4. Return the results in JSON format

            Return the results in JSON format.
            For any long-running operations like installations, please split them into smaller commands and check progress."""

            
            response = await self.agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                self.runnable_config
            )
            
            # Extract the content from the response
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            elif hasattr(response, "content"):
                return response.content
            else:
                print(f"Unexpected response format: {type(response)}")
                return {"error": "Invalid response format"}
            
        except Exception as e:
            print(f"Error in environment setup: {e}")
            return {"error": str(e)}
   
    async def benchmark_cycle(self):
        """Run one complete benchmark cycle with streaming output."""
        try:
            self.logger.start_benchmark()
            prompt = """Run a complete GPU benchmark cycle:
            0. First check, if we already have a GPU instance running
            1. If not, get a list of available GPUs and select the cheapest one
            2. Set up the environment with required dependencies
            3. Run performance tests
            4. Clean up resources when done
            
            Return the results in JSON format.
            For any long-running operations like installations, please split them into smaller commands and check progress."""

            # Create the runnable config with required keys
            runnable_config = RunnableConfig(
                recursion_limit=200,
                configurable={
                    "thread_id": "benchmark_cycle",
                    "checkpoint_ns": "benchmark",
                    "checkpoint_id": str(datetime.now().timestamp())
                }
            )

            print_system(f"\nStarted benchmark at: {datetime.now().strftime('%H:%M:%S')}")

            async for chunk in self.agent_executor.astream(
                {"messages": [HumanMessage(content=prompt)]},
                runnable_config
            ):
                if "agent" in chunk:
                    response = chunk["agent"]["messages"][0].content
                    print_ai(format_ai_message_content(response))
                    self.logger.log_message("assistant", response)
                elif "tools" in chunk:
                    system_msg = chunk["tools"]["messages"][0].content
                    print_system(system_msg)
                    self.logger.log_message("system", system_msg)

                print_system("-------------------")
            await self.logger.save_conversation()


        except KeyboardInterrupt:
            self.logger.set_status(BenchmarkStatus.INTERRUPTED, "User interrupted the benchmark")
            await self.logger.save_conversation()
            print_system("\nBenchmark interrupted...")
            await self._cleanup()
        except Exception as e:
            self.logger.set_status(BenchmarkStatus.ERROR, f"{type(e).__name__}: {str(e)}")
            await self.logger.save_conversation()
            print_error(f"Error during benchmark: {str(e)}")
            await self._cleanup()                   
        finally:
            await self._cleanup()

    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run intelligent performance benchmarks."""
        try:
            prompt = """Create a performance testing plan that includes:
            1. Memory bandwidth tests
            2. CUDA core utilization
            3. Mixed precision capabilities
            4. Power efficiency metrics
            
            Return a JSON with test configurations including:
            - test_name
            - commands: List of setup and test commands
            - metrics: What to measure
            - success_criteria: Performance thresholds
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            test_plan = json.loads(response.content)
            
            results = {}
            for test in test_plan:
                test_results = {}
                
                # Run setup commands
                for cmd in test["commands"]:
                    output = await self.tools["remote_shell"].arun({
                        "command": cmd
                    })
                    test_results[cmd] = output
                
                # Parse metrics from output
                metrics_prompt = f"""Extract the following metrics from this output:
                {json.dumps(test['metrics'])}
                
                Output:
                {output}
                
                Return only the JSON object with metric names and values."""
                
                metrics_response = await self.llm.ainvoke([HumanMessage(content=metrics_prompt)])
                metrics = json.loads(metrics_response.content)
                
                # Evaluate against success criteria
                test_results["metrics"] = metrics
                test_results["success"] = all(
                    metrics[metric] >= threshold 
                    for metric, threshold in test["success_criteria"].items()
                )
                
                results[test["test_name"]] = test_results
            
            return results
            
        except Exception as e:
            print(f"Error in performance tests: {e}")
            return {"error": str(e)}
   
        
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