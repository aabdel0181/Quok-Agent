import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Model for search results."""
    title: str
    content: str
    url: str
    source_type: str = "web"

class WebSearcher:
    """Simple class for performing web searches using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the web searcher with API key."""
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.logger = logging.getLogger(__name__)
        
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search for the query using Tavily API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        self.logger.info(f"Searching web for: {query}")
        
        if not self.api_key:
            self.logger.warning("No API key available for search. Please set TAVILY_API_KEY.")
            return []
        
        try:
            # Use Tavily API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.tavily.com/search",
                    json={
                        "query": query,
                        "max_results": num_results,
                        "api_key": self.api_key
                    }
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Search API returned status {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    for result in data.get("results", []):
                        results.append({
                            "title": result.get("title", "No title"),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "source_type": "web"
                        })
                    
                    return results
        
        except Exception as e:
            self.logger.error(f"Error in web search: {str(e)}")
            return [] 