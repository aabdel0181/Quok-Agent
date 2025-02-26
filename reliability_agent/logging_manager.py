from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json
import re
from utils import print_system, print_error
from langchain_community.chat_models import ChatAnthropic
from langchain_core.messages import HumanMessage

class BenchmarkStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    GRACEFUL_FAILURE = "GRACEFUL_FAILURE"
    ERROR = "ERROR"
    INTERRUPTED = "INTERRUPTED"
    UNDETERMINED = "UNDETERMINED"

class ConversationLogger:
    def __init__(self, base_log_dir="conversation_logs", llm=None):
        self.base_log_dir = base_log_dir
        self.conversation_log: List[Dict[str, str]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = BenchmarkStatus.UNDETERMINED
        self.error_context: Optional[str] = None
        self.last_activity_time: Optional[datetime] = None
        self.llm = llm or ChatAnthropic(model="claude-3-sonnet")
        self.metadata: Dict[str, Any] = {}

    async def analyze_conversation(self):
        """Use Claude to analyze the conversation log and determine the true status."""
        prompt = f"""Analyze this GPU benchmark conversation log and classify the outcome. 
    Return ONLY a JSON object (no other text) in this exact format:
    {{
        "status": "SUCCESS|PARTIAL_SUCCESS|GRACEFUL_FAILURE|ERROR",
        "confidence": 0.0,
        "reasoning": "brief explanation",
        "key_events": ["list of important events"],
        "recommendations": ["list of suggestions if any"]
    }}

    Conversation Log:
    {json.dumps(self.conversation_log, indent=2)}"""
        
        try:
            response = await self.llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            
            # Print raw response for debugging
            print_error(f"Raw response: {response.content}")
            
            # Clean up the response
            json_str = response.content.strip()
            # Find the first { and last }
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start >= 0 and end > 0:
                json_str = json_str[start:end]
                analysis = json.loads(json_str)
                self.status = BenchmarkStatus[analysis["status"]]
                self.error_context = analysis["reasoning"]
                self.metadata["claude_analysis"] = analysis
            else:
                print_error("No valid JSON found in response")
                
        except Exception as e:
            print_error(f"Error during log analysis: {e}")
    
    def add_metadata(self, key: str, value: Any):
        """Add additional metadata to the log."""
        self.metadata[key] = value

    def start_benchmark(self):
        """Start tracking a new benchmark run."""
        self.start_time = datetime.now()
        self.last_activity_time = self.start_time
        self.conversation_log = []
    def log_message(self, role: str, content: str):
        """Log a message with its role and content."""
        self.conversation_log.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity_time = datetime.now()
    def set_status(self, status: BenchmarkStatus, error_context: Optional[str] = None):
        """Set the benchmark status and optional error context."""
        self.status = status
        self.error_context = error_context
    def _extract_conversation_metrics(self) -> Dict[str, Any]:
        """Extract useful metrics from the conversation for RAG purposes."""
        metrics = {
            "message_count": len(self.conversation_log),
            "tool_calls": 0,
            "error_messages": 0
        }
        
        def get_content_as_string(message):
            content = message.get("content", "")
            if isinstance(content, list):
                return " ".join(str(item) for item in content)
            return str(content)

        word_count = 0
        for message in self.conversation_log:
            content = get_content_as_string(message)
            word_count += len(content.split())
            if message["role"] == "system" and "error" in content.lower():
                metrics["error_messages"] += 1
            if "Tool Call:" in content:
                metrics["tool_calls"] += 1
        
        metrics["total_words"] = word_count
        return metrics
    async def save_conversation(self):
        """Save the conversation log to a categorized directory structure."""
        if not self.conversation_log or not self.start_time:
            return

        # Analyze the conversation before saving
        # await self.analyze_conversation()

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        status_dir = os.path.join(self.base_log_dir, self.status.value.lower())
        os.makedirs(status_dir, exist_ok=True)
        filename = f"{status_dir}/benchmark_conversation_{timestamp}.json"
        
        metadata = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else datetime.now().isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "status": self.status.value,
            "error_context": self.error_context,
            "conversation": self.conversation_log,
            **self.metadata  # Include any additional metadata
        }

        metadata.update(self._extract_conversation_metrics())
        
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print_system(f"\nConversation log saved to: {filename}")