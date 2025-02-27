import os
import sys
import time
import asyncio
from datetime import datetime, timedelta
import random
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

class BenchmarkRunner:
    def __init__(self, 
                 min_interval_minutes=1, 
                 max_interval_minutes=3,
                 log_dir="benchmark_logs"):
        self.min_interval = timedelta(minutes=min_interval_minutes)
        self.max_interval = timedelta(minutes=max_interval_minutes)
        self.last_run = None
        self.setup_logging(log_dir)
        self.log_dir = Path(log_dir)
        
    def setup_logging(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"benchmark_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def save_run_output(self, run_id: str, stdout: str, stderr: str, success: bool):
        """Save detailed run output to a JSON file."""
        output_file = self.log_dir / f"run_{run_id}.json"
        output_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Saved detailed output to {output_file}")
        
    def get_next_interval(self) -> timedelta:
        minutes = random.uniform(
            self.min_interval.total_seconds() / 60,
            self.max_interval.total_seconds() / 60
        )
        return timedelta(minutes=minutes)

    async def stream_output(self, process: asyncio.subprocess.Process):
        """Stream output from the process in real-time."""
        stdout_chunks = []
        stderr_chunks = []

        async def read_stream(stream, chunks):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode()
                chunks.append(line_str)
                # Print in real-time
                print(line_str, end='')

        # Create tasks for both stdout and stderr
        if process.stdout:
            stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_chunks))
        if process.stderr:
            stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_chunks))

        # Wait for both streams to complete
        if process.stdout:
            await stdout_task
        if process.stderr:
            await stderr_task

        return ''.join(stdout_chunks), ''.join(stderr_chunks)

    async def run_single_benchmark(self):
        try:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            logging.info(f"Starting benchmark run {run_id}")

            # Run the run.py script
            process = await asyncio.create_subprocess_exec(
                'python3', 'run.py',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Stream and capture output
            stdout, stderr = await self.stream_output(process)
            
            # Wait for process to complete
            return_code = await process.wait()
            success = return_code == 0
            
            # Save detailed output
            self.save_run_output(run_id, stdout, stderr, success)
            
            if success:
                logging.info(f"Benchmark {run_id} completed successfully")
            else:
                logging.error(f"Benchmark {run_id} failed with return code {return_code}")
                
            # Log a summary of the output
            if stdout:
                logging.info("Last stdout lines:")
                last_lines = stdout.splitlines()[-5:]  # Last 5 lines
                for line in last_lines:
                    logging.info(f"  {line}")
            
            if stderr:
                logging.error("Last stderr lines:")
                last_lines = stderr.splitlines()[-5:]  # Last 5 lines
                for line in last_lines:
                    logging.error(f"  {line}")
                    
            return success
            
        except Exception as e:
            logging.error(f"Error in benchmark cycle: {e}", exc_info=True)
            return False

    async def run_continuous(self):
        logging.info("Starting continuous benchmark runner")
        
        while True:
            try:
                current_time = datetime.now()
                
                if self.last_run is None or (current_time - self.last_run) >= self.get_next_interval():
                    logging.info("Starting new benchmark cycle")
                    
                    success = await self.run_single_benchmark()
                    self.last_run = current_time
                    
                    next_interval = self.get_next_interval()
                    next_run = current_time + next_interval
                    
                    logging.info(
                        f"Benchmark cycle {'completed' if success else 'failed'}. "
                        f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"(in {next_interval.total_seconds() / 60:.1f} minutes)"
                    )
                
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logging.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in continuous run loop: {e}", exc_info=True)
                await asyncio.sleep(300)

def main():
    load_dotenv()
    runner = BenchmarkRunner()
    try:
        asyncio.run(runner.run_continuous())
    except KeyboardInterrupt:
        logging.info("Shutting down benchmark runner...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()