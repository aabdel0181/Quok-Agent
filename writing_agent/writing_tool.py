import os
import asyncio
import logging
from typing import Dict, List, Any, Optional

from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from .writing_agent import WritingAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WritingInput(BaseModel):
    """Input for the writing tool."""
    query: str = Field(description="The topic or request for content creation")
    reference_files: Optional[List[str]] = Field(
        description="Optional list of reference files for style analysis", 
        default=None
    )
    target_length: Optional[int] = Field(
        description="Target length of the article in words", 
        default=1500
    )
    output_file: Optional[str] = Field(
        description="Optional path to save the output text",
        default=None
    )

class WritingTool(BaseTool):
    """Tool for autonomous article writing with research capabilities."""
    
    name: str = "writing_agent"
    description: str = """
    A powerful tool for autonomous article writing with research capabilities.
    
    This tool can:
    - Analyze and mimic the writing style of reference documents
    - Research the web for up-to-date information
    - Create well-structured, informative articles
    - Generate citations and properly attribute sources
    - Save output to a specified file (optional)
    
    Examples of queries:
    - "Write an article about quantum computing"
    - "Create a blog post about climate change with a focus on recent developments"
    - "Write a technical overview of machine learning for beginners"
    - "Draft an article about the history of artificial intelligence citing major milestones"
    
    You can optionally provide:
    - Reference files to mimic their writing style
    - Target length for the article
    - Output file path to save the generated content
    """
    
    args_schema: type[BaseModel] = WritingInput
    api_key: Optional[str] = None
    
    def __init__(self, **kwargs):
        """Initialize the writing tool."""
        super().__init__(**kwargs)
        
        # Store API key for later use
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables.")
    
    async def _arun(self, query: str, reference_files: Optional[List[str]] = None, 
                    target_length: Optional[int] = 1500,
                    output_file: Optional[str] = None) -> str:
        """
        Run the writing agent asynchronously.
        
        Args:
            query: The topic or request for content creation
            reference_files: Optional list of reference files for style mimicry
            target_length: Target length of the article in words
            output_file: Optional path to save the output text
            
        Returns:
            Generated content as a string
        """
        try:
            print(f"Starting writing agent for topic: {query}")
            print(f"Target length: {target_length} words")
            
            # Initialize the agent
            agent = WritingAgent(api_key=self.api_key)
            
            # Always try to load reference materials
            success = await agent.load_reference_materials(reference_files)
            if not success:
                print("Warning: Failed to load reference materials")
                    
            # Generate content
            result = await agent.create_content(query, target_length, output_file)
            
            # Format the response
            if isinstance(result, dict):
                content = result.get("content", "")
                word_count = result.get("word_count", 0)
                sources = result.get("sources", [])
                
                # Add word count and sources info
                response = f"{content}\n\n---\nWord count: {word_count}\n"
                if sources:
                    response += "\nSources:\n"
                    for source in sources:
                        response += f"- {source}\n"
                        
                return response
            else:
                return str(result)
            
        except Exception as e:
            logger.error(f"Error in writing agent: {e}")
            return f"Error occurred while generating content: {str(e)}"
    
    def _run(self, query: str, reference_files: Optional[List[str]] = None,
             target_length: Optional[int] = 1500,
             output_file: Optional[str] = None) -> str:
        """Run the tool synchronously by delegating to the async implementation."""
        return asyncio.run(self._arun(
            query=query,
            reference_files=reference_files,
            target_length=target_length,
            output_file=output_file
        )) 