# Quok.it's Hyperbolic Reliability Agent

## Overview
The Quok.it Reliability Agent is a specialized component of the Hyperbolic AgentKit that automates GPU benchmarking and reliability testing. It systematically tests different GPU instances on the Hyperbolic platform, collecting performance metrics and generating reliability scores.

## Architecture

### Directory Structure
reliability_agent/
├── init.py
|── run.py # Single entrance function 
|── ec2-continuous.py # repeatedly selects new random gpus and runs run.py on them
├── agent.py # Main agent implementation

NEW TOOLS: 
hyperbolic_agentkit_core/actions/
- cubelas_benchmark_action.py # runs cublas benchmarking
- dynamodb_inserter.py # inserts error, benchmark, and health data into dynamodb
- gpu_health_check_action.py  # runs health checks and determines any areas of concern

### Core Components

1. **ReliabilityAgent (agent.py)**
   - Main agent class that orchestrates the benchmarking process
   - Handles GPU selection, rental, testing, and result collection
   - Integrates with Hyperbolic toolkit for GPU operations

2. **Custom Tools**
   - Allow the agent to take actions in the role of a GPU Quality Assurance Site Reliability Engineer (SRE)
     

## Workflow

1. **Initialization**
   ```python
   reliability_agent = ReliabilityAgent(hyperbolic_toolkit)
   ```

2. **Benchmark Cycle**
   ```mermaid
   graph TD
   A[Get Available GPUs] --> B[Select Random GPU]
   B --> C[Rent GPU Instance]
   C --> D[Run Health & Benchmark Suite]
   D --> E[Collect Metrics]
   E --> F[Calculate Score]
   F --> G[Store Results]
   G --> H[Terminate Instance]
   ```

3. **Data Collection**
   - GPU utilization
   - Memory usage
   - Temperature
   - Training performance
   - Error rates

4. **Scoring System**
   - Weighted performance metrics
   - Reliability indicators
   - Historical comparison

### NEW REQUIRED Environment Variables (in .env)

REPO_NAME=Quok-benchmark
REPO_URL=https://github.com/aabdel0181/Quok-benchmark.git
ACCESS_KEY=[DYNAMO_DB_ACCESS_KEY]
SECRET_ACCESS_KEY=[DYNAMO_DB_SECRET_ACCESS_KEY]
