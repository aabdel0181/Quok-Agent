from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json
import re
from utils import print_system

class BenchmarkStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    STALLED = "stalled"
    UNDETERMINED = "undetermined"

class ConversationLogger:
    def __init__(self, base_log_dir="conversation_logs"):
        self.base_log_dir = base_log_dir
        self.conversation_log: List[Dict[str, str]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = BenchmarkStatus.UNDETERMINED
        self.error_context: Optional[str] = None
        self.last_activity_time: Optional[datetime] = None

    def start_benchmark(self):
        """Initialize a new benchmark session."""
        self.start_time = datetime.now()
        self.last_activity_time = datetime.now()
        self.status = BenchmarkStatus.UNDETERMINED
        self.conversation_log = []
        
    def log_message(self, role: str, content: str):
        """Add a message to the conversation log."""
        self.conversation_log.append({"role": role, "content": content})
        self.last_activity_time = datetime.now()

    def set_status(self, status: BenchmarkStatus, error_context: Optional[str] = None):
        """Update the benchmark status."""
        self.status = status
        self.error_context = error_context
        self.end_time = datetime.now()

    def _extract_conversation_metrics(self) -> Dict[str, Any]:
        """Extract useful metrics from the conversation for RAG purposes."""
        metrics = {
            "message_count": len(self.conversation_log),
            "tool_calls": 0,
            "error_messages": 0,
            "keywords": set(),
        }
        
        error_patterns = [
            r"error",
            r"exception",
            r"failed",
            r"timeout",
            r"connection refused"
        ]
        
        keyword_patterns = [
            r"gpu",
            r"benchmark",
            r"test",
            r"performance",
            r"memory",
            r"cuda"
        ]
        
        def get_content_as_string(message):
            content = message.get("content", "")
            if isinstance(content, list):
                # If content is a list, join its elements
                return " ".join(str(item) for item in content)
            return str(content)
        metrics = {"total_messages": len(self.conversation_log)}
        word_count = 0
        for message in self.conversation_log:
            content = get_content_as_string(message)
            word_count += len(content.lower().split())
        
        metrics["total_words"] = word_count
        return metrics

    def save_conversation(self):
        """Save the conversation log to a categorized directory structure."""
        if not self.conversation_log or not self.start_time:
            return

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        status_dir = os.path.join(self.base_log_dir, self.status.value)
        os.makedirs(status_dir, exist_ok=True)
        filename = f"{status_dir}/benchmark_conversation_{timestamp}.json"
        
        metadata = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else datetime.now().isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "status": self.status.value,
            "error_context": self.error_context,
            "conversation": self.conversation_log
        }

        metadata.update(self._extract_conversation_metrics())
        
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print_system(f"\nConversation log saved to: {filename}")